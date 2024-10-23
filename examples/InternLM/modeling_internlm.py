#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange

# from flash_attn.layers.rotary import RotaryEmbedding
from flash_attn.modules.mha import FlashSelfAttention, FlashCrossAttention
from embedding import RotaryEmbedding

def scaled_init_method_normal(sigma: float = 1.0, num_layers: int = 1):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


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
    ):
        super().__init__()

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


class MHA(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        dropout: float = 0.0,
        softmax_scale: float = None,
        causal: bool = False,
        layer_idx: int = None,
        rotary_emb_dim: int = 0,
        rotary_emb_scale_base: int = 0,
        rope_base: int = 10000,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        qk_interleaved: Optional[bool] = True,
        enable_qkv_fusion: bool = True,
        out_bias: bool = True,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.causal = causal

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        self.kv_dim = self.head_dim * num_heads  # num_kv_heads equals to num_heads in MHA
        self.enable_qkv_fusion = enable_qkv_fusion

        self.rotary_emb_dim = rotary_emb_dim
        self.interleaved = qk_interleaved

        factory_kwargs = {"device": device, "dtype": dtype}

        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"

        if self.rotary_emb_dim > 0:
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rope_base,
                scale_base=rotary_emb_scale_base,
                device=device,
            )

        if self.enable_qkv_fusion:
            # bias=True is according to https://spaces.ac.cn/archives/9577
            self.wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias, **factory_kwargs)
        else:
            self.wq = nn.Linear(embed_dim, embed_dim, bias, **factory_kwargs)
            self.wk = nn.Linear(embed_dim, self.kv_dim, bias, **factory_kwargs)
            self.wv = nn.Linear(embed_dim, self.kv_dim, bias, **factory_kwargs)

        self.inner_attn = FlashSelfAttention(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)
        self.inner_cross_attn = FlashCrossAttention(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)

        # output projection always have the bias (for now) (except for baichuan2 model)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_bias, **factory_kwargs)

    def forward(self, x, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim)
        """
        # wqkv
        if self.enable_qkv_fusion:
            qkv = self.wqkv(x)
            qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, d=self.head_dim)

            # q = qkv[:, :, 0].squeeze(2)
            # k = qkv[:, :, 1].squeeze(2)
            # v = qkv[:, :, 2].squeeze(2)
        else:
            q, k, v = self.wq(x), self.wk(x), self.wv(x)
            q = rearrange(q, "b s (h d) -> b s h d", d=self.head_dim)
            k = rearrange(k, "b s (h d) -> b s h d", d=self.head_dim)
            v = rearrange(v, "b s (h d) -> b s h d", d=self.head_dim)

        # rotary embedding
        qkv = self.rotary_emb(qkv, **kwargs)
        kwargs.pop("indexes")

        qkv = qkv.squeeze(0)
        context = self.inner_attn(qkv, **kwargs)
        context = context.unsqueeze(0)

        # wo
        return self.out_proj(rearrange(context, "b s h d -> b s (h d)"))


class InternLM1Decoder(nn.Module):

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        mlp_ratio: int = 4,
        attn_drop_rate: float = 0,
        drop_rate: float = 0.0,
        dtype: torch.dtype = torch.float,
        layer_norm_epsilon: float = 1e-6,
        layer_idx: int = 0,
        residual_in_fp32: bool = False,
        device: Optional[torch.device] = None,
        qk_interleaved: bool = False,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        rope_base: int = 10000,
    ):
        super().__init__()
        # dropout selective checkpoint can only be enabled when checkpoint is disabled.
        self.layer_idx = layer_idx

        head_dim = hidden_size // num_attention_heads

        self.mixer = MHA(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            bias=True,
            dropout=attn_drop_rate,
            softmax_scale=1 / math.sqrt(head_dim),
            causal=True,
            layer_idx=layer_idx,
            rotary_emb_dim=head_dim,
            rotary_emb_scale_base=0,
            rope_base=rope_base,
            device=device,
            dtype=dtype,
            qk_interleaved=qk_interleaved,
            enable_qkv_fusion=True,
        )

        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)

        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon, device=device)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon, device=device)

        self.mlp = FeedForward(
            hidden_size,
            int(hidden_size * mlp_ratio),
            out_features=hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.use_swiglu = use_swiglu
        self.use_scaled_init = use_scaled_init
        self.residual_in_fp32 = residual_in_fp32  # only make sense when using prenorm
        self.return_residual = False
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.mixer.named_parameters():
                if param.ndim == 1:
                    param.data.zero_()
                elif "wqkv" in name:
                    normal_(std=0.006)(param.data)
                elif self.use_scaled_init:
                    scaled_init_method_normal(sigma=0.006, num_layers=self.layer_idx + 1)(param.data)
                else:
                    normal_(std=0.0015)(param.data)

            for name, param in self.mlp.named_parameters():
                if param.ndim == 1 and "bias" in name:
                    param.data.zero_()
                elif self.use_swiglu:
                    if self.use_scaled_init and "w2" in name:
                        scaled_init_method_normal(sigma=0.006, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        # candidate: w1, w3, fused_w1_w3
                        normal_(std=0.006 if "w1" in name or "w3" in name else 0.0015)(param.data)
                else:
                    if self.use_scaled_init and "fc1" not in name:
                        scaled_init_method_normal(sigma=0.006, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        normal_(std=0.006 if "fc1" in name else 0.0015)(param.data)


    def forward(self, hidden_states, **kwargs):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Attn/MLP(LN(residual))
            cu_seqlens: 1d LongTensor, len(cu_seqlens) = hidden_states + 1
            indexes: the length of index is same as hidden states, which stand for the current position
        """

        def _dropout_and_norm_attn(_hidden_states):
            _dropped = self.dropout1(_hidden_states)
            _residual = _dropped
            #FIXME: should set norm weight to torch.float32
            # _hidden_states = self.norm1(_residual.float())
            _hidden_states = self.norm1(_residual)
            return _residual, _hidden_states

        residual, hidden_states = _dropout_and_norm_attn(hidden_states)

        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(hidden_states, **kwargs)

        def _dropout_and_norm_ffn(_residual, _hidden_states):
            _dropped = self.dropout2(_hidden_states)
            _residual = (_dropped + _residual) if _residual is not None else _dropped
            #FIXME: should set norm weight to torch.float32
            # _hidden_states = self.norm2(_residual.float())
            _hidden_states = self.norm2(_residual)
            return _residual, _hidden_states

        residual, hidden_states = _dropout_and_norm_ffn(residual, hidden_states)

        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mlp(hidden_states)

        return hidden_states + residual


class InternLM1(nn.Module):

    def __init__(
        self,
        num_layers: int = 12,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        vocab_size: int = 50304,
        mlp_ratio: int = 4.0,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        dtype: torch.dtype = torch.float,
        layer_norm_epsilon: float = 1e-5,
        embed_grad_scale: float = 0.1,
        parallel_output: bool = True,
        start_layer_idx: int = 0,
        device: Optional[torch.device] = None,
        residual_in_fp32: bool = False,
        qk_interleaved: bool = False,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        rope_base: int = 10000,
    ):
        super().__init__()

        self.embed_grad_scale = embed_grad_scale
        self.parallel_output = parallel_output

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, device=device, dtype=dtype)

        for _, param in self.embedding.named_parameters():
            normal_(std=0.0052)(param)

        self.blocks = nn.ModuleList(
            [
                InternLM1Decoder(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    dtype=dtype,
                    layer_norm_epsilon=layer_norm_epsilon,
                    layer_idx=lid + start_layer_idx,  # This parameter is used for caching during generation
                    residual_in_fp32=residual_in_fp32,
                    device=device,
                    qk_interleaved=qk_interleaved,
                    use_scaled_init=use_scaled_init,
                    use_swiglu=use_swiglu,
                    rope_base=rope_base,
                )
                for lid in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon, device=device)
        self.head = nn.Linear(
            in_features=hidden_size,
            out_features=vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        for _, param in self.head.named_parameters():
            normal_(std=0.0052)(param)

    def forward(self, hidden_states=None, input_ids=None, **kwargs):
        # attention_mask: compute attention on the places where the value is 1
        if hasattr(self, "embedding") and input_ids is not None:
            hidden_states = self.embedding(input_ids)
            if self.embed_grad_scale != 1:
                hidden_states = (
                    self.embed_grad_scale * hidden_states + (1 - self.embed_grad_scale) * hidden_states.detach()
                )

        for _, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, **kwargs)

        if hasattr(self, "norm"):
            #FIXME: should set norm weight to torch.float32
            # hidden_states = self.norm(hidden_states.float())
            hidden_states = self.norm(hidden_states)
        if hasattr(self, "head"):
            hidden_states = self.head(hidden_states)

        return hidden_states
