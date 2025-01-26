
'''
modified from nn.Transformer
'''
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import xavier_uniform_
from typing import Any, Callable, Optional, Union
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.module import Module
from torch.nn.modules.normalization import LayerNorm
import torch.nn as nn
import torch
import copy

# 先制造一个静态的map，不是autoregressive的map


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_torch_dtype(dtype_str):
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }
    return dtype_map.get(dtype_str, None)


class DecoderOnlyTransformer(Module):
    def __init__(self, config):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")

        decoder_layer = DecoderOnlyTransformerLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation=config['activation'],
            layer_norm_eps=config['layer_norm_eps'],
            bias=config['bias'],
            device=torch.device(config['device']),
            dtype=_get_torch_dtype(config['dtype']),
        )
        self.layers = _get_clones(decoder_layer, config['num_layers'])
        self.num_layers = config['num_layers']

    def forward(self, tgt):
        output = tgt
        for mod in self.layers:
            output = mod(output)
        return output


class DecoderOnlyTransformerLayer(nn.Module):
    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            **factory_kwargs,
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(self, tgt):
        x = tgt
        x = self.norm1(x + self._self_attention_block(x))
        x = self.norm2(x + self._feed_forward_block(x))
        return x

    def _self_attention_block(self, x):
        # some parameters are not used in this function
        attn_mask = None
        key_padding_mask = None
        is_causal = False

        # forward function
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    def _feed_forward_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
