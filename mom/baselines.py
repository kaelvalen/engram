"""Parameter-matched baselines (spec §6.2).

    B1  ENGRAM fixed 3:1 SSD:GDR hybrid (the incumbent)
    B2  SSD-only        B3  GDR-only
    B4  MoM with frozen uniform routing (g = 1/K) — value of *learned* routing
    B5  MoM with seeded random per-token routing — value vs. arbitrary routing

B4/B5 are MoMLM instances with the router mode overridden; B1–B3 are stacks
of the corresponding ENGRAM residual blocks (mixer + conv + FFN), i.e. the
fixed-composition incumbent family.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn
from engram.modules.delta import DeltaBlock
from engram.modules.norm import RMSNorm
from engram.modules.ssd import SSDBlock

from .config import MoMConfig
from .model import MoMLM

BASELINE_KINDS = ("B1", "B2", "B3", "B4", "B5")


def layer_pattern(kind: str, num_layers: int) -> list[str]:
    """Fixed per-layer composition for the incumbent baselines."""
    if kind == "B1":  # ENGRAM 3:1 SSD:GDR — the special case MoM recovers (§1.2)
        return [("gdr" if (i + 1) % 4 == 0 else "ssd") for i in range(num_layers)]
    if kind == "B2":
        return ["ssd"] * num_layers
    if kind == "B3":
        return ["gdr"] * num_layers
    raise ValueError(f"no fixed pattern for baseline kind {kind!r}")


def _build_engram_block(name: str, cfg: MoMConfig) -> nn.Module:
    if name == "ssd":
        return SSDBlock(
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            state_dim=cfg.ssd_state_dim,
            dt_min=cfg.s4_dt_min,
            dt_max=cfg.s4_dt_max,
            conv_kernel_size=cfg.conv_kernel_size,
            ffn_expand=cfg.ffn_expand,
            scan_backend=cfg.scan_backend,
            dropout=cfg.dropout,
        )
    if name == "gdr":
        return DeltaBlock(
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            qk_norm=cfg.qk_norm,
            chunk_size=cfg.delta_chunk_size,
            gate_bias_init=cfg.gate_bias_init,
            conv_kernel_size=cfg.conv_kernel_size,
            ffn_expand=cfg.ffn_expand,
            backend=cfg.delta_backend,
            dropout=cfg.dropout,
        )
    raise ValueError(f"unknown block {name!r}")


class HybridLM(nn.Module):
    """Fixed-composition ENGRAM backbone + LM head (B1–B3)."""

    def __init__(self, kind: str, cfg: MoMConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.kind = kind
        self.pattern = layer_pattern(kind, cfg.num_layers)
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, cfg.hidden_dim)
        self.blocks = nn.ModuleList([_build_engram_block(t, cfg) for t in self.pattern])
        self.norm_f = RMSNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, states: list | None = None) -> dict:
        h = self.embed(input_ids)
        if states is None:
            states = [None] * len(self.blocks)
        new_states = []
        for block, st in zip(self.blocks, states):
            conv_state = st[0] if st is not None else None
            mixer_state = st[1] if st is not None else None
            h, new_conv, new_mixer = block(h, conv_state, mixer_state)
            new_states.append((new_conv, new_mixer))
        logits = self.lm_head(self.norm_f(h))
        return {"logits": logits, "routings": [], "states": new_states}

    def param_report(self) -> dict[str, int]:
        report = {"embedding": self.embed.weight.numel(), "lm_head": self.lm_head.weight.numel()}
        for i, (name, block) in enumerate(zip(self.pattern, self.blocks)):
            report[f"layer{i}.{name}"] = sum(p.numel() for p in block.parameters())
        report["total"] = sum(p.numel() for p in self.parameters())
        return report


def build_model(kind: str, cfg: MoMConfig, vocab_size: int) -> nn.Module:
    """Single entry point for MoM and every registered baseline."""
    if kind == "mom":
        return MoMLM(cfg, vocab_size)
    if kind in ("B1", "B2", "B3"):
        return HybridLM(kind, cfg, vocab_size)
    if kind == "B4":
        return MoMLM(replace(cfg, router_mode="uniform"), vocab_size)
    if kind == "B5":
        return MoMLM(replace(cfg, router_mode="random"), vocab_size)
    raise ValueError(f"unknown model kind {kind!r} (mom | {BASELINE_KINDS})")
