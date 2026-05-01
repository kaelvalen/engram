from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModalityConfig


class PerModalityHead(nn.Module):
    """Per-modality classification head.
    
    Independent Linear(hidden_dim -> num_classes) for each modality.
    Backbone shared, head modality-specific.
    """

    def __init__(self, modalities: list[ModalityConfig], hidden_dim: int):
        super().__init__()
        self.heads = nn.ModuleDict({
            m.name: nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, m.num_classes),
            )
            for m in modalities
        })

    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        # x: [B, hidden_dim]
        return self.heads[modality](x)
