from typing import Optional, Union

import torch

from einops import rearrange
from rotary_emb import apply_rotary as _flash_apply_rotary_func


class ApplyRotaryEmb(torch.autograd.Function):
    """
    ApplyRotaryEmb
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        interleaved: bool = False,
        in_place: bool = False,
        use_fused_rope: bool = True,
    ):
        """
            x: (batch_size, seqlen, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding to the first rotary_dim of x.
        """
        *_, seqlen, _, head_dim = x.shape
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2

        assert rotary_dim <= head_dim
        assert seqlen <= rotary_seqlen
        assert sin.shape == (rotary_seqlen, rotary_dim // 2)

        x_ro = x[..., :rotary_dim]
        x1, x2 = (x_ro[..., ::2], x_ro[..., 1::2]) if interleaved else x_ro.chunk(2, dim=-1)

        if in_place:
            out, o1, o2 = x, x1, x2
        else:
            out = torch.empty_like(x)
            out_ro = out[..., :rotary_dim]
            o1, o2 = (out_ro[..., ::2], out_ro[..., 1::2]) if interleaved else out_ro.chunk(2, dim=-1)

        _flash_apply_rotary_func(
            x1,
            x2,
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            o1,
            o2,
            False,
        )

        if rotary_dim < head_dim and not in_place:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])

        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        ctx.in_place = in_place
        ctx.use_fused_rope = use_fused_rope

        return out

    @staticmethod
    def backward(ctx, do):
        cos, sin = ctx.saved_tensors
        *_, seqlen, _, head_dim = do.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2

        do_ro = do[..., :rotary_dim]
        do1, do2 = (do_ro[..., ::2], do_ro[..., 1::2]) if ctx.interleaved else do_ro.chunk(2, dim=-1)

        if ctx.in_place:
            dx, dx1, dx2 = do, do1, do2
        else:
            dx = torch.empty_like(do)
            dx_ro = dx[..., :rotary_dim]
            dx1, dx2 = (dx_ro[..., ::2], dx_ro[..., 1::2]) if ctx.interleaved else dx_ro.chunk(2, dim=-1)

        _flash_apply_rotary_func(
            do1,
            do2,
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            dx1,
            dx2,
            True,
        )

        if rotary_dim < head_dim and not ctx.in_place:
            dx[..., rotary_dim:].copy_(do[..., rotary_dim:])

        return dx, None, None, None, None

def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False, in_place: bool = False
):
   
    return ApplyRotaryEmb.apply(x, cos, sin, interleaved, in_place)

class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base > 0, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(self, dim: int, base=10000, scale_base=0, device=None):
        """ """
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        self.scale_base = scale_base
        self.scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base > 0
            else None
        )

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _update_cos_sin_cache(
        self, x: torch.Tensor, indexes: Union[int, torch.Tensor] = 0, max_seqlen: Optional[int] = None
    ):
        """x: (batch, seqlen, nheads, headdim)"""
        if max_seqlen is not None:
            seqlen = max_seqlen
        elif isinstance(indexes, int):
            seqlen = indexes + x.shape[1]
        else:
            # Note that this statement may cause synchronization between CPU and GPU,
            # so it's best to precompute and pass in max_seqlen ahead of time
            seqlen = indexes.max().item()

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seqlen > self._seq_len_cached or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=x.device, dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(x.dtype)
                self._sin_cached = torch.sin(freqs).to(x.dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)

    def _get_slice(self, tensor: torch.Tensor, offsets: Union[int, torch.Tensor] = 0):
        if isinstance(offsets, int):
            return tensor[offsets:]
        else:
            return tensor[offsets]

    def _convert_padding(
        self, x: torch.Tensor, empties: torch.Tensor, convert_type: str = "left2right", in_place: bool = False
    ):
        # TODO: impl in_place = True.
        assert not in_place, "in_place = True is NYI."
        assert convert_type in ("left2right", "right2left"), f"Unknown convert type {convert_type}"

        ret = x.clone()

        for i in range(len(empties)):
            if empties[i] == 0:
                continue

            if convert_type == "left2right":
                ret[i][: -empties[i]] = x[i][empties[i] :]
                ret[i][-empties[i] :] = x[i][: empties[i]]
            else:  # right2left
                ret[i][empties[i] :] = x[i][: -empties[i]]
                ret[i][: empties[i]] = x[i][-empties[i] :]

        return ret

    def forward(
        self,
        x: torch.Tensor,
        offsets: Union[int, torch.Tensor] = 0,
        max_seqlen: Optional[int] = None,
        cache_type: str = "query",
        interleaved: bool = False,
        in_place: bool = False,
        left_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Applies rotary position embeddings to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            offsets (Union[int, torch.Tensor], optional): The sequence offsets for the input. Defaults to 0.
            max_seqlen (Optional[int], optional): The maximum sequence length for caching. Defaults to None.
            cache_type (str, optional): Specifies whether the cache is for 'query' or 'key'. Defaults to "query".
            interleaved (bool, optional): Whether the input tensor is interleaved. Defaults to False.
            in_place (bool, optional): Whether the operation should be done in-place. Defaults to False.
            left_padding_mask (Optional[torch.Tensor], optional): A mask for left padding. Defaults to None.

        Returns:
            torch.Tensor: The tensor with applied rotary position embeddings.
        """
        assert cache_type in ("query", "key"), f"Unknown cache type {cache_type}"
        assert isinstance(offsets, (int, torch.Tensor)), f"Invalid offsets type {type(offsets)}"

        if left_padding_mask is not None:
            empties = left_padding_mask[..., -1].sum(dim=-1)
            x = self._convert_padding(x, empties, convert_type="left2right", in_place=in_place)

        self._update_cos_sin_cache(x, offsets, max_seqlen)

        cos_cached = self._cos_k_cached if cache_type == "key" and self.scale is not None else self._cos_cached
        sin_cached = self._sin_k_cached if cache_type == "key" and self.scale is not None else self._sin_cached
        ret = apply_rotary_emb(
            x, self._get_slice(cos_cached, offsets), self._get_slice(sin_cached, offsets), interleaved, in_place
        )

        if left_padding_mask is not None:
            ret = self._convert_padding(ret, empties, convert_type="right2left", in_place=in_place)

        return ret
