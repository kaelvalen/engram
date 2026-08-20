"""MoM language model: token embedding + MoM blocks + LM head."""

from __future__ import annotations

import torch
import torch.nn as nn
from engram.modules.norm import RMSNorm

from .block import MoMBlock
from .config import MoMConfig
from .router import RoutingOutput
from .state import ExpertStateDict


class MoMLM(nn.Module):
    """Causal LM over a stack of MoM blocks (spec §4 configs)."""

    def __init__(self, cfg: MoMConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, cfg.hidden_dim)
        self.blocks = nn.ModuleList([MoMBlock(cfg, i) for i in range(cfg.num_layers)])
        self.norm_f = RMSNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: ExpertStateDict | None = None,
        knockout: dict[int, set[str]] | None = None,
    ) -> dict:
        """input_ids: [B, T] long. Returns logits, per-layer routing, states.

        ``knockout`` maps layer index → expert names to force-exclude
        (analysis §7.3); routers renormalise over the survivors.
        """
        h = self.embed(input_ids)
        routings: list[RoutingOutput] = []
        new_states = ExpertStateDict()
        for i, block in enumerate(self.blocks):
            exclude = knockout.get(i) if knockout else None
            h, updates, routing = block(h, states, exclude=exclude)
            new_states.update(updates)
            routings.append(routing)
        logits = self.lm_head(self.norm_f(h))
        return {"logits": logits, "routings": routings, "states": new_states}

    def empty_state(self, batch_size: int, device, dtype) -> ExpertStateDict:
        """Zero-initialised global state (spec §3.6)."""
        out = ExpertStateDict()
        for block in self.blocks:
            out.update(block.empty_states(batch_size, device, dtype))
        return out

    def param_report(self) -> dict[str, int]:
        """Parameter budget per component, per spec §3.2's reporting rule."""
        report = {"embedding": self.embed.weight.numel(), "lm_head": self.lm_head.weight.numel()}
        for i, block in enumerate(self.blocks):
            for name, expert in block.experts.items():
                report[f"layer{i}.{name}"] = sum(p.numel() for p in expert.parameters())
            report[f"layer{i}.router"] = sum(p.numel() for p in block.router.parameters())
        report["total"] = sum(p.numel() for p in self.parameters())
        return report
