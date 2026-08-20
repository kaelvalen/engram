"""Expert registry (spec §3.2, §8).

Maps expert names to the ENGRAM mixer implementations.  ENGRAM blocks are
consumed, never modified beyond the additive write-mask flags (§12.1).
"swa" is scaffolded for v2 and raises NotImplementedError when built (§1.4).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from engram.modules.attention import SlidingWindowAttention, SWAState
from engram.modules.delta import GatedDeltaRule
from engram.modules.ssd import SSDMixer

EXPERT_NAMES: tuple[str, ...] = ("ssd", "gdr", "swa")


def build_expert(name: str, cfg) -> nn.Module:
    """Instantiate an expert mixer with ENGRAM-default hyperparameters."""
    if name == "ssd":
        return SSDMixer(
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            state_dim=cfg.ssd_state_dim,
            dt_min=cfg.s4_dt_min,
            dt_max=cfg.s4_dt_max,
            scan_backend=cfg.scan_backend,
        )
    if name == "gdr":
        return GatedDeltaRule(
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            qk_norm=cfg.qk_norm,
            chunk_size=cfg.delta_chunk_size,
            gate_bias_init=cfg.gate_bias_init,
            backend=cfg.delta_backend,
        )
    if name == "swa":
        # v2 expert, activated now that ENGRAM's SWA has a streaming KV-cache
        # with the MoM masked-execution semantics (spec §3.4 SWA bullet).
        return SlidingWindowAttention(cfg.hidden_dim, cfg.num_heads, window=cfg.swa_window)
    raise ValueError(f"unknown expert {name!r}; registered: {EXPERT_NAMES}")


def expert_forward(
    name: str,
    expert: nn.Module,
    x: torch.Tensor,
    state,
    write_mask: torch.Tensor | None,
    cfg,
):
    """Uniform expert contract: (x, state_in, mask) -> (y, state_out).

    Applies the spec's masked-execution semantics per expert kind (§3.4):
    SSD decays on every step unless ``cfg.decay_on_skip`` is False (D1);
    GDR's α forget gate is neutralised on a miss unless
    ``cfg.gdr_decay_on_skip`` is True (spec-exact pass-through default).
    """
    if name == "ssd":
        return expert(x, state, write_mask=write_mask, freeze_on_mask=not cfg.decay_on_skip)
    if name == "gdr":
        return expert(x, state, write_mask=write_mask, freeze_on_mask=not cfg.gdr_decay_on_skip)
    if name == "swa":
        # Window slides over the routed subsequence; non-routed tokens are
        # excluded from the window (pass-through is exact by construction).
        return expert(x, state, write_mask=write_mask)
    raise ValueError(f"unknown expert {name!r}")


def expert_empty_state(name: str, expert: nn.Module, batch_size: int, device, dtype):
    """Zero-initialised streaming state (spec §3.6)."""
    if name == "swa":
        return SWAState(
            k=expert.empty_state(batch_size, device, dtype).k,
            v=expert.empty_state(batch_size, device, dtype).v,
            pos=torch.zeros(batch_size, dtype=torch.long, device=device),
        )
    return expert.empty_state(batch_size, device, dtype)


def expert_param_count(expert: nn.Module) -> int:
    """Parameter budget per expert (§3.2 reporting requirement)."""
    return sum(p.numel() for p in expert.parameters())
