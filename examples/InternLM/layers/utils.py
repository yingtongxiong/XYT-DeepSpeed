import math

import torch
from torch import nn, Tensor


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


def convert_attn_args_to_kwargs(args, kwargs):

    if len(args) == 0:
        return kwargs

    assert len(args) == 3, "args must be generate by convert_attn_kwargs_to_args function"

    if args[0] is not None:
        assert "cu_seqlens" not in kwargs, "repeated 'cu_seqlens' argument exists both in args and kwargs"
        kwargs["cu_seqlens"] = args[0]
    if args[1] is not None:
        assert "indexes" not in kwargs, "repeated 'indexes' argument exists both in args and kwargs"
        kwargs["indexes"] = args[1]
    if args[2] is not None:
        assert "max_seqlen" not in kwargs, "repeated 'max_seqlen' argument exists both in args and kwargs"
        kwargs["max_seqlen"] = args[2]

    return kwargs


def dropout_and_norm_residual(
    norm_class,
    dropout_class,
    hidden_states,
    residual=None,
    checkpoint=False,
    dtype=None,
    residual_in_fp32=False,
    activation_offload=False,
    use_reentrant=True,
):
    """
    Wrapper for dropout and normalization process with activation checkpoint.

    If residual is None: norm_class(dropout_class(hidden_states))
    Otherwise: norm_class(dropout_class(hidden_states) + residual)
    """
    if dtype is None:
        dtype = hidden_states.dtype

    def _dropout_and_norm_call(_residual, _hidden_states):
        _dropped = dropout_class(_hidden_states)

        if _residual is None:
            _residual = _dropped
        else:
            _residual = _residual + _dropped

        _hidden_states = norm_class(_residual.to(dtype))

        return _residual, _hidden_states

    residual, hidden_states = _dropout_and_norm_call(residual, hidden_states)

    if residual_in_fp32:
        residual = residual.to(torch.float32)

    return residual, hidden_states