from __future__ import annotations

import torch
import torch.nn as nn


class _ResBlock1D(nn.Module):
    def __init__(self, ch: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(ch, ch, kernel_size, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)) + x)


class ResNet1DClassifier(nn.Module):
    """Light 1D ResNet over time for ``[B, T, C]`` ECG-like inputs."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 64,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[_ResBlock1D(base_channels) for _ in range(num_blocks)])
        self.head = nn.Linear(base_channels, num_classes)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
        # x: [B, T, C] -> conv1d expects [B, C, T]
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.blocks(x)
        x = x.mean(dim=-1)
        logits = self.head(x)
        out: dict = {"logits": logits}
        if labels is not None:
            out["loss"] = nn.functional.cross_entropy(logits, labels)
        return out
