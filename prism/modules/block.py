from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .attention import SWABlock
from .delta import DeltaBlock, DeltaState
from .s4 import S4Block
from .ssd import SSDBlock


@dataclass
class BlockState:
    """Tek bir katmanın carry state'i."""

    conv_state: torch.Tensor | None  # [B, dim, kernel_size-1] (None for attention)
    mixer_state: torch.Tensor | DeltaState | None  # SSM tensor | DeltaState | KV-cache


def build_block(layer_type: str, cfg) -> nn.Module:
    """Config'e göre blok üret. 's4' rolü ssm_kind'e göre SSD veya S4D olur."""
    if layer_type == "s4":
        if cfg.ssm_kind == "ssd":
            return SSDBlock(
                hidden_dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                state_dim=cfg.ssd_state_dim,
                dt_min=cfg.s4_dt_min,
                dt_max=cfg.s4_dt_max,
                conv_kernel_size=cfg.conv_kernel_size,
                ffn_expand=cfg.ffn_expand,
                scan_backend=cfg.scan_backend,
            )
        return S4Block(
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            state_mult=cfg.s4_state_mult,
            dt_min=cfg.s4_dt_min,
            dt_max=cfg.s4_dt_max,
            conv_kernel_size=cfg.conv_kernel_size,
            ffn_expand=cfg.ffn_expand,
            init=cfg.s4d_init,
            scan_backend=cfg.scan_backend,
        )
    elif layer_type == "delta":
        return DeltaBlock(
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            qk_norm=cfg.qk_norm,
            chunk_size=cfg.delta_chunk_size,
            gate_bias_init=cfg.gate_bias_init,
            conv_kernel_size=cfg.conv_kernel_size,
            ffn_expand=cfg.ffn_expand,
            backend=cfg.delta_backend,
        )
    elif layer_type == "swa":
        return SWABlock(
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            window=cfg.swa_window,
            ffn_expand=cfg.ffn_expand,
        )
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")


def forward_block(
    block: nn.Module,
    layer_type: str,
    x: torch.Tensor,
    state: BlockState | None,
) -> tuple[torch.Tensor, BlockState]:
    """Tip-bağımsız blok forward — model.py bunu çağırır.

    Tüm bloklar (S4/SSD/Delta/SWA) aynı imzayı paylaşır:
        (x, conv_state, mixer_state) -> (x, new_conv_state, new_mixer_state)
    """
    conv_state = state.conv_state if state is not None else None
    mixer_state = state.mixer_state if state is not None else None
    x, new_conv, new_mixer = block(x, conv_state, mixer_state)
    return x, BlockState(conv_state=new_conv, mixer_state=new_mixer)
