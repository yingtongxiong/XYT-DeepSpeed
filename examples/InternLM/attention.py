from torch import nn

from flash_attn.flash_attn_interface import (
    flash_attn_varlen_kvpacked_func as _flash_varlen_kvpacked_func,
)

from flash_attn.flash_attn_interface import (
    flash_attn_varlen_func as _flash_varlen_qkvsplited_func,
)

class SelfAttention(nn.Module):
    """Implements scaled dot-product attention with optional softmax scaling.

    This class implements the scaled dot-product attention mechanism, which can be optionally scaled
    by a softmax scaling factor. It supports configurations for causal attention and applies dropout
    to the attention scores.

    Arguments:
        causal (bool): If True, applies causal attention to mask future tokens. Defaults to False.
        softmax_scale (Optional[float]): Scaling factor for attention scores before applying softmax.
            Defaults to 1/sqrt(d_keys) where d_keys is the dimension of the keys, computed at runtime.
        attention_dropout (float): Dropout rate for attention scores. Defaults to 0.0.
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0, layer_idx=0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = nn.Dropout(attention_dropout)
        self.layer_idx = layer_idx

    
    def _qkvsplited_forward(
        self,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=None,
        causal=False,
    ):
        softmax_scale = self.softmax_scale if softmax_scale is None else softmax_scale
        causal = self.causal if causal is None else causal


        dropout =  self.dropout.p

        # compatible data format: [1, packelen, 3, n_head, headim]
        q, k, v = q.squeeze(dim=0), k.squeeze(dim=0), v.squeeze(dim=0)

        # input_idxs: 0: q, 1: k, 2: v
        output = _flash_varlen_qkvsplited_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout,
            softmax_scale,
            causal,
        )

        return output.unsqueeze(dim=0)
    

    def forward(
        self,
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=None,
        causal=None,
    ):
        softmax_scale = self.softmax_scale if softmax_scale is None else softmax_scale
        causal = self.causal if causal is None else causal

        dropout = self.dropout.p
        
        q = q.squeeze(0)
        kv = kv.squeeze(0)

        output =  _flash_varlen_kvpacked_func(
            q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout, softmax_scale, causal,
        )
        
        return output.unsqueeze(dim=0)

    