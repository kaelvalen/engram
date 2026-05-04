from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .config import ModalityConfig, PRISMConfig
from .modules.block import BlockState, build_block, forward_block

logger = logging.getLogger(__name__)


class ModalityProjection(nn.Module):
    """Independent Linear(input_dim -> hidden_dim) projection for each modality."""

    def __init__(self, modalities: list[ModalityConfig], hidden_dim: int):
        super().__init__()
        self.projections = nn.ModuleDict(
            {m.name: nn.Linear(m.input_dim, hidden_dim, bias=False) for m in modalities}
        )

    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        return self.projections[modality](x)


class PRISMBackbone(nn.Module):
    """Modality-agnostic S4+Delta interleaved backbone."""

    def __init__(self, cfg: PRISMConfig):
        super().__init__()
        self.cfg = cfg
        self.pattern = cfg.layer_pattern()
        self.blocks = nn.ModuleList([build_block(t, cfg) for t in self.pattern])

    def forward(
        self,
        x: torch.Tensor,  # [B, T, hidden_dim]
        states: list[BlockState | None] | None = None,
    ) -> tuple[torch.Tensor, list[BlockState]]:
        if states is None:
            states = [None] * len(self.blocks)

        new_states = []
        for block, layer_type, state in zip(self.blocks, self.pattern, states):
            x, new_state = forward_block(block, layer_type, x, state)
            new_states.append(new_state)

        return x, new_states


class PerModalityHead(nn.Module):
    """Independent classifier head for each modality."""

    def __init__(self, modalities: list[ModalityConfig], hidden_dim: int):
        super().__init__()
        self.heads = nn.ModuleDict(
            {m.name: nn.Linear(hidden_dim, m.num_classes) for m in modalities}
        )

    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        return self.heads[modality](x)


class PRISMForClassification(nn.Module):
    """
    Full PRISM model for classification tasks.

    Forward:
        x        : [B, T, input_dim]  — raw signal / patch sequence
        modality : str                — target modality name
        labels   : [B] optional       — calculate CrossEntropy loss

    Returns:
        dict(logits, loss?)
    """

    def __init__(self, cfg: PRISMConfig):
        super().__init__()
        self.cfg = cfg
        self.projection = ModalityProjection(cfg.modalities, cfg.hidden_dim)
        self.backbone = PRISMBackbone(cfg)
        self.head = PerModalityHead(cfg.modalities, cfg.hidden_dim)
        self.norm = nn.ModuleDict({m.name: nn.LayerNorm(cfg.hidden_dim) for m in cfg.modalities})

    def forward(
        self,
        x: torch.Tensor,
        modality: str,
        labels: torch.Tensor | None = None,
        states: list[BlockState | None] | None = None,
    ) -> dict:
        # validate modality and input shape
        mcfg = next((m for m in self.cfg.modalities if m.name == modality), None)
        if mcfg is None:
            raise KeyError(
                f"Unknown modality '{modality}'. Registered: {[m.name for m in self.cfg.modalities]}"
            )
        if x.shape[-1] != mcfg.input_dim:
            raise ValueError(
                f"Input last dim {x.shape[-1]} does not match "
                f"expected {mcfg.input_dim} for modality '{modality}'"
            )

        # 1. projection
        x = self.projection(x, modality)  # [B, T, hidden_dim]

        # 2. backbone
        x, new_states = self.backbone(x, states)  # [B, T, hidden_dim]

        # 3. explicit pooling strategy (mean pooling vs last token)
        if self.cfg.pool_type == "mean":
            x = x.mean(dim=1)  # [B, hidden_dim]
        elif self.cfg.pool_type == "last":
            x = x[:, -1, :]  # [B, hidden_dim]

        # 4. norm + head
        x = self.norm[modality](x)
        logits = self.head(x, modality)  # [B, num_classes]

        out = {"logits": logits, "states": new_states}

        if labels is not None:
            out["loss"] = nn.functional.cross_entropy(logits, labels)

        return out
