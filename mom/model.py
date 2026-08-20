"""MoM language model: token embedding + MoM blocks + LM head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        (analysis §7.3); routers renormalise over the survivors. When the
        per-layer surprise predictor is enabled (cfg.use_surprise_predictor),
        an auxiliary ``pred_loss`` (MSE of each predictor's online head against
        that block's own input, stop-grad) is accumulated and returned in the
        dict — only in training mode, so eval stays clean.
        """
        h = self.embed(input_ids)
        routings: list[RoutingOutput] = []
        new_states = ExpertStateDict()
        pred_loss = torch.zeros((), device=input_ids.device)
        for i, block in enumerate(self.blocks):
            exclude = knockout.get(i) if knockout else None
            if (
                self.training
                and self.cfg.use_surprise_predictor
                and block.surprise_predictor is not None
            ):
                # Per-block input is `h` right now; online head predicts it from
                # h_{t-1}, target is the block input detached (JEPA-style).
                pl = block.surprise_predictor.predict_online(h)
                pred_loss = pred_loss + F.mse_loss(pl, h.detach())
            h, updates, routing = block(h, states, exclude=exclude)
            new_states.update(updates)
            routings.append(routing)
        logits = self.lm_head(self.norm_f(h))
        return {
            "logits": logits,
            "routings": routings,
            "states": new_states,
            "pred_loss": pred_loss,
        }

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
