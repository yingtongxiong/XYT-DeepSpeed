from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def Silu(w1_o, w2_o):
    return F.silu(w1_o) * w2_o


class FeedForward(nn.Module):
    """
    Base FeedForward in flash implementation.

    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        multiple_of (int): For efficient training. Reset the size of hidden feature. 256 by default.
        mlp_layer_fusion (Optional[Bool]):  Some linears without bias in FFN can be fused to reduce the comm cost of SP.
        activation_type (str): the activation function used for feed forward, "swiglu" by default.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        multiple_of: int = 256,
        mlp_layer_fusion: Optional[bool] = False,
        activation_type: str = "swiglu",
    ):
        super().__init__()

        # TODO: support gelu...
        assert activation_type in ("swiglu"), f"Unsupported activation type: {activation_type}"

        self.mlp_layer_fusion = mlp_layer_fusion

        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)

        if self.mlp_layer_fusion:
            assert bias is False, "Fuesd FeedForward only support bias is False."

            self.fused_w1_w3 = nn.Linear(
                in_features, hidden_features * 2, bias, device=device, dtype=dtype
            )
            self.w2 = nn.Linear(hidden_features, out_features, bias, device=device, dtype=dtype)

        else:
            self.w1 = nn.Linear(in_features, hidden_features, bias, device=device, dtype=dtype)
            self.w3 = nn.Linear(in_features, hidden_features, bias, device=device, dtype=dtype)
            self.w2 = nn.Linear(hidden_features, out_features, bias, device=device, dtype=dtype)

    def forward(self, x):
        if not self.mlp_layer_fusion:
            w1_o = self.w1(x)
            w3_o = self.w3(x)
        else:
            fussed_out = self.fused_w1_w3(x)
            w1_o, w3_o = torch.split(fussed_out, fussed_out.shape[-1] // 2, dim=-1)
        out = self.w2(Silu(w1_o, w3_o))
        return out