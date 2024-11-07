import math
from typing import Optional

import torch
from einops import rearrange
from torch import nn


from .embedding import RotaryEmbedding
from .norm import RMSNorm
from .attention import SelfAttention



class GQA(nn.Module):
    """
    Multi-head self-attention and cross-attention.

    Args:
        embed_dim (int): The dimention of hidden state.
        num_heads (int): The number of attention heads.
        num_kv_heads (int): The number of attention heads for key and value.
        max_position_embeddings (int): max position embeddings, 2048 by default.
        bias (bool): Whether the bias is needed for linears. Will be used when initializing QKV matrix and
                     output projection. False by default.
        dropout (float): The dropout rate for cross attention and self attention. 0.0 by default.
        causal (boolean): Whether to apply causal attention mask. False by default.
        layer_idx (int): The index of current layer. None by default.
        use_dynamic_ntk_rope (bool): whether use dynamic ntk rope, false by default.
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        rotary_emb_scale_base (int): The scaling factor of Rotary Embedding. If scale_base > 0, this implements
                                    XPos(Sun et al., https://arxiv.org/abs/2212.10554). 0 by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        qk_interleaved (Optional[bool]): whether the odd and even columns of wq and wk is interleaved. True by default.
        enable_qkv_fusion (bool): whether wq, wk and wv lienar is fused. True by default.
        enable_qk_norm (bool): whether to apply QK-norm for training stabilization. False by default.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int = None,
        max_position_embeddings: int = 2048,
        bias: bool = False,
        dropout: float = 0.0,
        causal: bool = False,
        layer_idx: int = None,
        use_dynamic_ntk_rope: bool = False,
        rope_base: int = 10000,
        rotary_emb_scale_base: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        qk_interleaved: Optional[bool] = True,
        qk_norm_type: str = "rmsnorm",
        layer_norm_epsilon: float = 1e-6,
        enable_qkv_fusion: bool = True,
        enable_qk_norm: bool = False,
        rope_scaling_factor: float = 1.0,
        rope_scaling: Optional[dict] = None,
        sequence_parallel: bool = False,
        sequence_world_size: int = 1,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.causal = causal

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim if head_dim else self.embed_dim // num_heads
        self.enable_qkv_fusion = enable_qkv_fusion

        self.softmax_scale = 1 / math.sqrt(self.head_dim)

        self.need_dkv_reduction = False
        if not sequence_parallel:
            self.num_kv_heads = max(num_kv_heads, sequence_world_size)
            self.need_dkv_reduction = num_kv_heads < sequence_world_size
        self.q_dim = self.head_dim * self.num_heads
        self.kv_dim = self.head_dim * self.num_kv_heads
        self.q_per_kv = num_heads // self.num_kv_heads

        self.use_dynamic_ntk_rope = use_dynamic_ntk_rope
        self.rotary_emb_dim = self.head_dim
        self.max_position_embeddings = max_position_embeddings
        self.interleaved = qk_interleaved

        factory_kwargs = {"device": device, "dtype": dtype}

        assert self.use_dynamic_ntk_rope is False, "Not support dynamic ntk rope yet."
        assert self.embed_dim % num_heads == 0, "embedding dim must be divisible by num_heads"

        if self.rotary_emb_dim > 0:
            self.rotary_emb = RotaryEmbedding(
                dim=self.rotary_emb_dim,
                base=rope_base,
                scale_base=rotary_emb_scale_base,
                device=device,
            )

        if enable_qkv_fusion:
            self.wqkv = nn.Linear(embed_dim, self.q_dim + 2 * self.kv_dim, bias, **factory_kwargs)
        else:
            self.wq = nn.Linear(embed_dim, self.q_dim, bias, **factory_kwargs)
            self.wk = nn.Linear(embed_dim, self.kv_dim, bias, **factory_kwargs)
            self.wv = nn.Linear(embed_dim, self.kv_dim, bias, **factory_kwargs)

        self.enable_qk_norm = enable_qk_norm
        if enable_qk_norm:
            if torch.distributed.get_rank() == 0:
                print("Enable qk norm.", flush=True)
            self.q_norm = RMSNorm(self.q_dim, eps=layer_norm_epsilon)
            self.k_norm = RMSNorm(self.kv_dim, eps=layer_norm_epsilon)

        self.inner_attn = SelfAttention(
            causal=causal, softmax_scale=self.softmax_scale, attention_dropout=dropout, layer_idx=layer_idx
        )
        # self.inner_cross_attn = CrossAttention(
        #     causal=causal, softmax_scale=self.softmax_scale, attention_dropout=dropout, layer_idx=layer_idx
        # )

        self.wo = nn.Linear(self.q_dim, embed_dim, bias, **factory_kwargs)


    def _apply_qkv_transform(self, x: torch.Tensor):
        """
        Transform input tensor into Query (Q), Key (K), and Value (V) tensors for attention.
        Supports both fused and separate QKV transform with optional QK-norm.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, embedding_dim]

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing (query, key, value) tensors
        """
        if self.enable_qkv_fusion:
            qkv = self.wqkv(x)
            qkv = rearrange(qkv, "b s (h gs d) -> b s h gs d", gs=self.q_per_kv + 2, d=self.head_dim)
            q, k, v = (qkv[..., : self.q_per_kv, :], qkv[..., -2, :], qkv[..., -1, :])

            if self.enable_qk_norm:
                q = rearrange(q, "b s h gs d -> b s (h gs d)")
                q = self.q_norm(q).to(q.dtype)
                q = rearrange(q, "b s (h gs d) -> b s (h gs) d", gs=self.q_per_kv, d=self.head_dim)

                k = rearrange(k, "b s h d -> b s (h d)")
                k = self.k_norm(k).to(k.dtype)
                k = rearrange(k, "b s (h d) -> b s h d", d=self.head_dim)
            else:
                q = rearrange(q, "b s h gs d -> b s (h gs) d")
        else:
            q, k, v = self.wq(x), self.wk(x), self.wv(x)
            if self.enable_qk_norm:
                q = self.q_norm(q).to(q.dtype)
                k = self.k_norm(k).to(k.dtype)
            q = rearrange(q, "b s (h d) -> b s h d", d=self.head_dim)
            k = rearrange(k, "b s (h d) -> b s h d", d=self.head_dim)
            v = rearrange(v, "b s (h d) -> b s h d", d=self.head_dim)

        return q, k, v

    def forward(self, x, inference_params=None, **kwargs):
        if inference_params is None:
            return self._training(x=x, **kwargs)
        else:
            return self._inference(x=x, inference_params=inference_params, **kwargs)

    def _training(self, x, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim)
        """
        q, k, v = self._apply_qkv_transform(x)

        # kwargs = _convert_cu_seqlens_for_qksplited(kwargs)

        # rotary embedding
        if self.rotary_emb_dim > 0:
            indexes = kwargs.pop("indexes", 0)
            max_seqlen_q = kwargs.get("max_seqlen", None)
            max_seqlen_k = kwargs.get("max_seqlen", None)

            q = self.rotary_emb(
                q, offsets=indexes, max_seqlen=max_seqlen_q, cache_type="query", interleaved=self.interleaved
            )
            k = self.rotary_emb(
                k, offsets=indexes, max_seqlen=max_seqlen_k, cache_type="key", interleaved=self.interleaved
            )

        kv = torch.concat([k.unsqueeze(2), v.unsqueeze(2)], dim=2)

        # if gpc.config.data.use_packed_dataset is False:
        kwargs["max_seqlen_q"] = kwargs["max_seqlen"]
        kwargs["max_seqlen_k"] = kwargs["max_seqlen"]
        kwargs["cu_seqlens_q"] = kwargs["cu_seqlens"]
        kwargs["cu_seqlens_k"] = kwargs["cu_seqlens"]
        kwargs.pop("cu_seqlens")
        kwargs.pop("max_seqlen")

        # self attention
        context = self.inner_attn(q, kv, **kwargs)

        # wo
        return self.wo(rearrange(context, "b s h d -> b s (h d)"))

    def _convert_unpacked_qkv_to_packed(
        self, q: torch.Tensor, kv: torch.Tensor, batch_size: int, attention_mask: torch.Tensor
    ):
        cu_seqlens = torch.concat(
            [
                torch.tensor([0], dtype=torch.int32, device=attention_mask.device),
                attention_mask.sum(dim=-1).to(dtype=torch.int32),
            ],
            dim=0,
        ).cumsum(dim=0, dtype=torch.int32)

        cu_seqlens_q = cu_seqlens
        cu_seqlens_k = cu_seqlens

        max_seqlen_q = attention_mask.shape[-1]
        max_seqlen_k = attention_mask.shape[-1]

        q_packed = (
            q.masked_select(attention_mask.view(batch_size, -1, 1, 1)).view(-1, q.shape[-2], q.shape[-1]).unsqueeze(0)
        )
        kv_packed = (
            # TODO consider window_size
            kv.masked_select(attention_mask.view(batch_size, -1, 1, 1, 1))
            .view(-1, kv.shape[-3], kv.shape[-2], kv.shape[-1])
            .unsqueeze(0)
        )

        return q_packed, kv_packed, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
