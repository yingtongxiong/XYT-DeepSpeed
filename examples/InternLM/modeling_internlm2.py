# Copyright (c) InternLM. All rights reserved.
import math
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange

# from flash_attn.layers.rotary import RotaryEmbedding
from flash_attn.modules.mha import FlashSelfAttention, FlashCrossAttention
from embedding import RotaryEmbedding
from attention import SelfAttention



def normal_(mean: float = 0.0, std: float = 1.0):
    r"""Return the initializer filling the input Tensor with values drawn from the normal distribution

     .. math::
        \mathcal{N}(\text{mean}, \text{std}^2)

    Args:
        mean (float): the mean of the normal distribution. Defaults 0.0.
        std (float): the standard deviation of the normal distribution. Defaults 1.0.
    """

    def initializer(tensor: Tensor):
        return nn.init.normal_(tensor, mean, std)

    return initializer


def uniform_(mean: float = 0.0, std: float = 1.0):
    r"""Return the initializer filling the input Tensor with values drawn from the uniform distribution

     .. math::
        \mathcal{U}(mean-a, mean+a), where a satisfies \mathcal{U}_{std}=std.

    Args:
        mean (float): the mean of the uniform distribution. Defaults 0.0.
        std (float): the standard deviation of the uniform distribution. Defaults 1.0.
    """

    a = math.sqrt(3.0 * std)

    def initializer(tensor: Tensor):
        return nn.init.uniform_(tensor, mean - a, mean + a)

    return initializer


def scaled_init_method_uniform(sigma: float = 1.0, num_layers: int = 1):
    """Init method based on p(x)=Uniform(-a, a) where std(x)=sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)
    a = math.sqrt(3.0 * std)

    def init_(tensor):
        return nn.init.uniform_(tensor, -a, a)

    return init_


def scaled_init_method_normal(sigma: float = 1.0, num_layers: int = 1):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return nn.init.normal_(tensor, mean=0.0, std=std)

    return init_



def Silu(w1_o, w2_o):
    return F.silu(w1_o) * w2_o

class FeedForward(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        multiple_of: int = 256,
    ):
        super().__init__()
        
        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            in_features, hidden_features, bias, device=device, dtype=dtype
        )
        self.w2 = nn.Linear(
            hidden_features, out_features, bias, device=device, dtype=dtype
        )
        self.w3 = nn.Linear(
            in_features, hidden_features, bias, device=device, dtype=dtype
        )

    def forward(self, x):
        w1_o = self.w1(x)
        w3_o = self.w3(x)
        out = self.w2(Silu(w1_o, w3_o))
        return out
    
class GQA(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 2048,
        head_dim: int = None,
        bias: bool = False,
        dropout: float = 0.0,
        softmax_scale: float = None,
        causal: bool = False,
        layer_idx: int = None,
        use_dynamic_ntk_rope: bool = False,
        rope_base: int = 10000,
        rotary_emb_dim: int = 0,
        rotary_emb_scale_base: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        qk_interleaved: Optional[bool] = True,
        enable_qkv_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.causal = causal

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        if head_dim:
            self.head_dim = head_dim
            q_dim = head_dim * num_heads
        else:
            self.head_dim = self.embed_dim // num_heads
            q_dim = embed_dim
        self.num_kv_heads = num_kv_heads
        self.q_per_kv = num_heads // num_kv_heads
        self.kv_dim = self.head_dim * num_kv_heads
        self.enable_qkv_fusion = enable_qkv_fusion

        self.use_dynamic_ntk_rope = use_dynamic_ntk_rope
        self.rotary_emb_dim = rotary_emb_dim
        self.max_position_embeddings = max_position_embeddings
        self.interleaved = qk_interleaved

        factory_kwargs = {"device": device, "dtype": dtype}

        assert self.use_dynamic_ntk_rope is False, "Not support dynamic ntk rope yet."
        assert self.embed_dim % num_heads == 0, "embedding dim must be divisible by num_heads"

        if self.rotary_emb_dim > 0:
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rope_base,
                scale_base=rotary_emb_scale_base,
                device=device,
            )
            
        if enable_qkv_fusion:
            assert bias is False, "Fuesd wqkv only support bias is False."
            self.wqkv = nn.Linear(embed_dim, q_dim + 2 * self.kv_dim, bias, **factory_kwargs)
        else:
            self.wq = nn.Linear(embed_dim, q_dim, bias, **factory_kwargs)
            self.wk = nn.Linear(embed_dim, self.kv_dim, bias, **factory_kwargs)
            self.wv = nn.Linear(embed_dim, self.kv_dim, bias, **factory_kwargs)

        self.inner_attn = SelfAttention(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        self.inner_cross_attn = FlashCrossAttention(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )

        self.wo = nn.Linear(q_dim, embed_dim, bias, **factory_kwargs)


    def forward(self, x, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim)
        """
        # wqkv
        if self.enable_qkv_fusion:
            qkv = self.wqkv(x)
            qkv = rearrange(qkv, "b s (h gs d) -> b s h gs d", gs=self.q_per_kv + 2, d=self.head_dim)
            q, k, v = (qkv[..., : self.q_per_kv, :], qkv[..., -2, :], qkv[..., -1, :])
            q = rearrange(q, "b s h gs d -> b s (h gs) d")
        else:
            q, k, v = self.wq(x), self.wk(x), self.wv(x)
            q = rearrange(q, "b s (h d) -> b s h d", d=self.head_dim)
            k = rearrange(k, "b s (h d) -> b s h d", d=self.head_dim)
            v = rearrange(v, "b s (h d) -> b s h d", d=self.head_dim)


        # rotary embedding
        if self.rotary_emb_dim > 0:
            indexes = kwargs.pop("indexes", 0)

            q = self.rotary_emb(
                q, offsets=indexes, max_seqlen=4096, cache_type="query", interleaved=self.interleaved
            )
            k = self.rotary_emb(
                k, offsets=indexes, max_seqlen=4096, cache_type="key", interleaved=self.interleaved
            )
            
        kv = torch.concat([k.unsqueeze(2), v.unsqueeze(2)], dim=2)

        kwargs["max_seqlen_q"] = kwargs["max_seqlen"]
        kwargs["max_seqlen_k"] = kwargs["max_seqlen"]
        kwargs["cu_seqlens_q"] = kwargs["cu_seqlens"]
        kwargs["cu_seqlens_k"] = kwargs["cu_seqlens"]
        kwargs.pop("cu_seqlens")
        kwargs.pop("max_seqlen")

        # self attention
        # q, kv = q.squeeze(dim=0), kv.squeeze(dim=0)
        context = self.inner_attn(q, kv, **kwargs)

        # qkv = qkv.squeeze(0)
        # context = self.inner_attn(qkv, **kwargs)
        # context = context.unsqueeze(0)

        # wo
        return self.wo(rearrange(context, "b s h d -> b s (h d)"))

    

class InternLM2Decoder(nn.Module):

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_kv_attention_heads: int = 12,
        mlp_ratio: int = 4,
        attn_drop_rate: float = 0,
        drop_rate: float = 0.0,
        max_position_embeddings: int = 2048,
        dtype: torch.dtype = torch.float,
        layer_norm_epsilon: float = 1e-6,
        checkpoint: bool = False,
        layer_idx: int = 0,
        use_dynamic_ntk_rope: bool = False,
        residual_in_fp32: bool = False,
        device: Optional[torch.device] = None,
        apply_post_layer_norm: bool = False,
        fused_dropout_add_ln: bool = True,
        no_bias: bool = False,
        norm_type: str = "rmsnorm",
        qk_interleaved: bool = False,
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        attn_wqkv_init_std: float = 0.02,
        attn_other_init_std: float = 0.02,
        ffn_uplayer_init_std: float = 0.02,
        ffn_other_init_std: float = 0.02,
        init_type: str = "normal",
        rope_base: int = 10000,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 256,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        # dropout selective checkpoint can only be enabled when checkpoint is disabled.
        self.dropout_selective_checkpoint = dropout_selective_checkpoint is True and checkpoint is False
        self.layer_idx = layer_idx
        self.prenorm = not apply_post_layer_norm
        assert not fused_dropout_add_ln, "dropout_add_layer_norm can not be used here"
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.attn_wqkv_init_std = attn_wqkv_init_std
        self.attn_other_init_std = attn_other_init_std
        self.ffn_uplayer_init_std = ffn_uplayer_init_std
        self.ffn_other_init_std = ffn_other_init_std

        self.max_position_embeddings = max_position_embeddings
        self.use_dynamic_ntk_rope = use_dynamic_ntk_rope

        head_dim = hidden_size // num_attention_heads
        self.attention = GQA(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_attention_heads,
            dropout=attn_drop_rate,
            max_position_embeddings=max_position_embeddings,
            softmax_scale=1 / math.sqrt(head_dim),
            causal=True,
            layer_idx=layer_idx,
            use_dynamic_ntk_rope=use_dynamic_ntk_rope,
            rotary_emb_dim=head_dim,
            rotary_emb_scale_base=0,
            device=device,
            dtype=dtype,
            qk_interleaved=qk_interleaved,
            bias=not no_bias,
            rope_base=rope_base,
            enable_qkv_fusion=True,
        )

        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)
        self.attention_norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon, device=device)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon, device=device)

        self.feed_forward = FeedForward(
            hidden_size,
            int(hidden_size * mlp_ratio),
            out_features=hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
            multiple_of=multiple_of,
        )

        self.use_swiglu = use_swiglu
        self.use_scaled_init = use_scaled_init
        self.residual_in_fp32 = residual_in_fp32  # only make sense when using prenorm
        self.return_residual = False

        if init_type == "normal":
            self.init_func = normal_
            self.scaled_init_func = scaled_init_method_normal
        else:
            self.init_func = uniform_
            self.scaled_init_func = scaled_init_method_uniform

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.attention.named_parameters():
                if param.ndim == 1:
                    param.data.zero_()
                elif "wq" in name or "wk" in name or "wv" in name:
                    self.init_func(std=self.attn_wqkv_init_std)(param.data)
                elif self.use_scaled_init:  # wo
                    self.scaled_init_func(sigma=self.attn_other_init_std, num_layers=self.layer_idx + 1)(param.data)
                else:
                    self.init_func(std=self.attn_other_init_std)(param.data)

            for name, param in self.feed_forward.named_parameters():
                if self.use_swiglu:
                    if self.use_scaled_init and "w2" in name:
                        self.scaled_init_func(sigma=self.ffn_other_init_std, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        # candidate: w1, w3, fused_w1_w3
                        self.init_func(
                            std=self.ffn_uplayer_init_std if "w1" in name or "w3" in name else self.ffn_other_init_std
                        )(param.data)
                else:
                    if self.use_scaled_init and "fc1" not in name:
                        self.scaled_init_func(sigma=self.ffn_other_init_std, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        self.init_func(std=self.ffn_uplayer_init_std if "fc1" in name else self.ffn_other_init_std)(
                            param.data
                        )

    def forward(self, hidden_states, residual, **kwargs):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Attn/MLP(LN(residual))
            cu_seqlens: 1d LongTensor, len(cu_seqlens) = hidden_states + 1
            indexes: the length of index is same as hidden states, which stand for the current position
        """
        if self.prenorm:

            def _dropout_and_norm_attn(_residual, _hidden_states):
                _dropped = self.dropout1(_hidden_states)
                _residual = (_dropped + _residual) if _residual is not None else _dropped
                _hidden_states = self.attention_norm(_residual.to(dtype=self.attention_norm.weight.dtype))

                return _residual, _hidden_states

            residual, hidden_states = _dropout_and_norm_attn(residual, hidden_states)

            if self.residual_in_fp32:
                residual = residual.to(torch.float32)

            hidden_states = self.attention(hidden_states, **kwargs)

            if not isinstance(self.feed_forward, nn.Identity):
                if not self.fused_dropout_add_ln:

                    def _dropout_and_norm_ffn(_residual, _hidden_states):
                        _dropped = self.dropout2(_hidden_states)
                        _residual = (_dropped + _residual) if _residual is not None else _dropped
                        _hidden_states = self.ffn_norm(_residual.to(self.ffn_norm.weight.dtype))

                        return _residual, _hidden_states

 
                    residual, hidden_states = _dropout_and_norm_ffn(residual, hidden_states)

                    if self.residual_in_fp32:
                        residual = residual.to(torch.float32)
                hidden_states = self.feed_forward(hidden_states)

            return hidden_states + residual
        else:
            assert residual is None

            mixer_out = self.attention(hidden_states, **kwargs)
            if self.return_residual:  # mixer out is actually a pair here
                mixer_out, hidden_states = mixer_out
            hidden_states = self.attention_norm(self.dropout1(mixer_out) + hidden_states).to(
                dtype=self.attention_norm.weight.dtype
            )
            if not isinstance(self.feed_forward, nn.Identity):
                mlp_out = self.feed_forward(hidden_states)
                if self.return_residual:  # mlp out is actually a pair here
                    mlp_out, hidden_states = mlp_out
                hidden_states = self.ffn_norm((self.dropout2(mlp_out)) + hidden_states).to(
                    dtype=self.ffn_norm.weight.dtype
                )
            return hidden_states


class InternLM2(nn.Module):

    def __init__(
        self,
        num_layers: int = 48,
        hidden_size: int = 2048,
        num_attention_heads: int = 32,
        num_kv_attention_heads: int = 32,
        vocab_size: int = 50304,
        mlp_ratio: float = 4.0,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        max_position_embeddings: int = 2048,
        dtype: torch.dtype = torch.float,
        checkpoint: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        first: bool = False,
        last: bool = False,
        embed_grad_scale: float = 0.1,
        parallel_output: bool = True,
        start_layer_idx: int = 0,
        use_dynamic_ntk_rope: bool = False,
        device: Optional[torch.device] = None,
        apply_post_layer_norm=False,
        no_bias=False,
        residual_in_fp32: bool = False,
        norm_type: str = "rmsnorm",
        qk_interleaved: bool = False,
        is_reward: bool = False,
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        embedding_init_std: float = 0.02,
        attn_wqkv_init_std: float = 0.02,
        attn_other_init_std: float = 0.02,
        ffn_uplayer_init_std: float = 0.02,
        ffn_other_init_std: float = 0.02,
        out_head_init_std: float = 0.02,
        init_type: str = "normal",
        rope_base: int = 10000,
        norm_head: bool = False,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 256,
    ):
        super().__init__()

        checkpoint_layer_num = int(num_layers * checkpoint)
        self.embed_grad_scale = embed_grad_scale
        self.parallel_output = parallel_output

        if first:
            self.tok_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)

            for _, param in self.tok_embeddings.named_parameters():
                if init_type == "normal":
                    normal_(std=embedding_init_std)(param)
                else:
                    uniform_(std=embedding_init_std)(param)

        self.layers = nn.ModuleList(
            [
                InternLM2Decoder(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_kv_attention_heads=num_kv_attention_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    max_position_embeddings=max_position_embeddings,
                    dtype=dtype,
                    layer_norm_epsilon=layer_norm_epsilon,
                    checkpoint=lid < checkpoint_layer_num,
                    layer_idx=lid + start_layer_idx,  # This parameter is used for caching during generation
                    use_dynamic_ntk_rope=use_dynamic_ntk_rope,
                    residual_in_fp32=residual_in_fp32,
                    device=device,
                    apply_post_layer_norm=apply_post_layer_norm,
                    fused_dropout_add_ln=False,
                    no_bias=no_bias,
                    norm_type=norm_type,
                    dropout_selective_checkpoint=dropout_selective_checkpoint,
                    use_scaled_init=use_scaled_init,
                    use_swiglu=use_swiglu,
                    qk_interleaved=qk_interleaved,
                    attn_wqkv_init_std=attn_wqkv_init_std,
                    attn_other_init_std=attn_other_init_std,
                    ffn_uplayer_init_std=ffn_uplayer_init_std,
                    ffn_other_init_std=ffn_other_init_std,
                    init_type=init_type,
                    rope_base=rope_base,
                    mlp_layer_fusion=mlp_layer_fusion,
                    multiple_of=multiple_of,
                )
                for lid in range(num_layers)
            ]
        )

        if last:
            if not apply_post_layer_norm:
                self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon, device=device)
                self.norm = self.norm.to(torch.float32)

            self.output = nn.Linear(
                in_features=hidden_size,
                out_features=vocab_size,
                bias=False,
                device=device,
                dtype=dtype,
            )
            for _, param in self.output.named_parameters():
                if init_type == "normal":
                    normal_(std=out_head_init_std)(param)
                else:
                    uniform_(std=out_head_init_std)(param)

    def forward(self, hidden_states=None, input_ids=None, **kwargs):
        # attention_mask: compute attention on the places where the value is 1
        if hasattr(self, "tok_embeddings") and input_ids is not None:
            hidden_states = self.tok_embeddings(input_ids)
            if self.embed_grad_scale != 1:
                hidden_states = (
                    self.embed_grad_scale * hidden_states + (1 - self.embed_grad_scale) * hidden_states.detach()
                )

        for _, block in enumerate(self.layers):
            hidden_states = block(hidden_states, residual=None, **kwargs)

        if hasattr(self, "norm"):
            hidden_states = self.norm(hidden_states.to(self.norm.weight.dtype))
        if hasattr(self, "output"):
            hidden_states = self.output(hidden_states)

        return hidden_states


    