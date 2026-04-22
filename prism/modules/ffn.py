from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network.
    
    hidden = hidden_dim * expand
    out = SiLU(gate) * up → down
    """

    def __init__(self, dim: int, expand: int = 2):
        super().__init__()
        hidden = dim * expand
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj   = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )
