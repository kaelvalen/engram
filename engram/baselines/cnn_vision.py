"""Compact 2D CNN baseline for CIFAR-10 classification.

Parameter-matched (~250k parameters) to provide a fair, apples-to-apples
comparison with engram_image (hidden_dim=64, num_layers=4).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _unpatchify(
    x: torch.Tensor, patch_size: int = 4, img_size: int = 32, in_ch: int = 3
) -> torch.Tensor:
    """Convert [B, num_patches, patch_size*patch_size*in_ch] -> [B, in_ch, img_size, img_size]."""
    if x.ndim == 4:
        return x
    B, N, D = x.shape
    hp = wp = img_size // patch_size
    # [B, hp, wp, patch_size, patch_size, in_ch]
    x = x.view(B, hp, wp, patch_size, patch_size, in_ch)
    # [B, in_ch, hp, patch_size, wp, patch_size] -> [B, in_ch, H, W]
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, in_ch, img_size, img_size)
    return x


class _ResBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + res)


class CompactConvNet2D(nn.Module):
    """Parameter-matched (~250k) 2D CNN baseline for CIFAR-10.

    Accepts patchified input [B, N, D] or direct 2D images [B, C, H, W].
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        channels: tuple[int, ...] = (32, 48, 64, 96),
        patch_size: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
        )

        stages = []
        in_c = channels[0]
        for idx, out_c in enumerate(channels):
            stride = 1 if idx == 0 else 2
            stages.append(_ResBlock2D(in_c, out_c, stride=stride))
            in_c = out_c

        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(channels[-1], num_classes)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        modality: str | None = None,
    ) -> dict[str, torch.Tensor]:
        if x.ndim == 3:
            x = _unpatchify(x, patch_size=self.patch_size)
        feat = self.stem(x)
        feat = self.stages(feat)
        feat = self.pool(feat).flatten(1)
        logits = self.head(feat)

        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            out["loss"] = nn.functional.cross_entropy(logits, labels)
        return out
