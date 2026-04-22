from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .delta import DeltaBlock, DeltaState
from .s4 import S4Block


@dataclass
class BlockState:
    """Tek bir katmanın carry state'i."""
    conv_state:  torch.Tensor        # [B, dim, kernel_size-1]
    mixer_state: torch.Tensor | DeltaState  # S4: complex tensor | Delta: DeltaState


def build_block(layer_type: str, cfg) -> nn.Module:
    """Config'e göre S4Block veya DeltaBlock üret."""
    if layer_type == "s4":
        return S4Block(
            hidden_dim      = cfg.hidden_dim,
            num_heads       = cfg.num_heads,
            state_mult      = cfg.s4_state_mult,
            dt_min          = cfg.s4_dt_min,
            dt_max          = cfg.s4_dt_max,
            conv_kernel_size= cfg.conv_kernel_size,
            ffn_expand      = cfg.ffn_expand,
        )
    elif layer_type == "delta":
        return DeltaBlock(
            hidden_dim      = cfg.hidden_dim,
            num_heads       = cfg.num_heads,
            qk_norm         = cfg.qk_norm,
            chunk_size      = cfg.delta_chunk_size,
            gate_bias_init  = cfg.gate_bias_init,
            conv_kernel_size= cfg.conv_kernel_size,
            ffn_expand      = cfg.ffn_expand,
        )
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")


def forward_block(
    block: nn.Module,
    layer_type: str,
    x: torch.Tensor,
    state: BlockState | None,
) -> tuple[torch.Tensor, BlockState]:
    """Tip-bağımsız blok forward — model.py bunu çağırır."""
    conv_state  = state.conv_state  if state is not None else None
    mixer_state = state.mixer_state if state is not None else None

    if layer_type == "s4":
        x, new_conv, new_mixer = block(x, conv_state, mixer_state)
    else:
        x, new_conv, new_mixer = block(x, conv_state, mixer_state)

    return x, BlockState(conv_state=new_conv, mixer_state=new_mixer)
