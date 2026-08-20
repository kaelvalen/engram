from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticMelPatchDataset(Dataset):
    """Pseudo–mel patch sequences for smoke tests and default audio training without files.

    Each item is ``[num_frames, mel_bins]`` (treated like a 1D signal over time).
    """

    def __init__(
        self, *, length: int, num_frames: int, mel_bins: int, num_classes: int, seed: int = 0
    ):
        super().__init__()
        self.length = length
        self.num_frames = num_frames
        self.mel_bins = mel_bins
        self.num_classes = num_classes
        g = torch.Generator().manual_seed(seed)
        self.feats = torch.randn(length, num_frames, mel_bins, generator=g)
        self.labels = torch.randint(0, num_classes, (length,), generator=g)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.feats[idx], self.labels[idx]


def get_audio_loaders(
    root: str = "./datasets/audio",
    batch_size: int = 32,
    num_mel_bins: int = 64,
    patch_frames: int = 4,
    num_workers: int = 0,
    *,
    synthetic: bool = True,
    train_size: int = 2048,
    val_size: int = 512,
    num_classes: int = 10,
) -> tuple[DataLoader, DataLoader]:
    """Return train/val loaders for audio modality.

    When ``synthetic`` is True (default), ignores ``root`` and returns in-memory data.
    When False, expects precomputed ``.pt`` tensors under ``root`` (optional future format).
    """
    num_frames = max(32, patch_frames * 32)

    if synthetic:
        train_ds = SyntheticMelPatchDataset(
            length=train_size,
            num_frames=num_frames,
            mel_bins=num_mel_bins,
            num_classes=num_classes,
            seed=1,
        )
        val_ds = SyntheticMelPatchDataset(
            length=val_size,
            num_frames=num_frames,
            mel_bins=num_mel_bins,
            num_classes=num_classes,
            seed=2,
        )
    else:
        root_path = Path(root)
        train_path = root_path / "train.pt"
        val_path = root_path / "val.pt"
        if not train_path.is_file() or not val_path.is_file():
            raise FileNotFoundError(
                f"Expected {train_path} and {val_path} when synthetic=False. "
                "Use --audio-synthetic or prepare tensor dumps."
            )
        train_ds = torch.load(train_path, weights_only=False)
        val_ds = torch.load(val_path, weights_only=False)

    pin = torch.cuda.is_available()
    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
        ),
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
        ),
    )
