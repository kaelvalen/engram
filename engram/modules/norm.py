from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Bias yok, sadece scale — transformer'larda standart hale geldi.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        # Accumulate in float32 for bf16/fp16 stability, but keep float64 as
        # float64 so fp64 gradcheck / equivalence tests stay exact (SGMS §9).
        acc = x if x.dtype in (torch.float32, torch.float64) else x.float()
        norm = acc.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (acc * norm).to(x.dtype) * self.weight


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """L2 normalization — QK-norm için."""
    return F.normalize(x, p=2, dim=dim, eps=eps)
