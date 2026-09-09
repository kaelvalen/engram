"""Simple 1D CNN baseline for Speech Commands classification.

Inspired by M5 (Dai et al., "Very deep convolutional neural networks for raw
waveforms") adapted for mel-spectrogram patch sequences [B, T, D].
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class AudioCNNClassifier(nn.Module):
    """Lightweight 1D CNN for audio classification over [B, T, D] patch sequences.

    Architecture: stem (projection) → 4 conv blocks with progressive channel
    expansion → global average pool → classifier head.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        base_channels: int = 64,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, base_channels, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
        )
        layers: list[nn.Module] = []
        ch = base_channels
        for i in range(num_blocks):
            out_ch = ch * 2 if i in (1, 3) else ch
            layers.append(_ConvBlock1D(ch, out_ch, kernel_size=7))
            ch = out_ch
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Linear(ch, num_classes)

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None, modality: str | None = None
    ) -> dict:
        # x: [B, T, D] → conv1d expects [B, D, T]
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.blocks(x)
        x = x.mean(dim=-1)
        logits = self.head(x)
        out: dict = {"logits": logits}
        if labels is not None:
            out["loss"] = nn.functional.cross_entropy(logits, labels)
        return out
