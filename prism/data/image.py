from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def patchify(x: torch.Tensor, patch_size: int = 4) -> torch.Tensor:
    """CIFAR görüntüsünü patch sequence'e çevir.
    
    x: [B, C, H, W]
    returns: [B, num_patches, patch_size*patch_size*C]
    """
    B, C, H, W = x.shape
    P = patch_size
    assert H % P == 0 and W % P == 0

    x = x.reshape(B, C, H // P, P, W // P, P)
    x = x.permute(0, 2, 4, 3, 5, 1)          # [B, H/P, W/P, P, P, C]
    x = x.reshape(B, (H // P) * (W // P), P * P * C)
    return x


class PatchCollator:
    """DataLoader collate fn — batch'i patchify eder."""

    def __init__(self, patch_size: int = 4):
        self.patch_size = patch_size

    def __call__(self, batch):
        imgs, labels = zip(*batch)
        imgs   = torch.stack(imgs)              # [B, C, H, W]
        labels = torch.tensor(labels)
        imgs   = patchify(imgs, self.patch_size)  # [B, N, patch_dim]
        return imgs, labels


def get_cifar_loaders(
    root: str = "./datasets/cifar",
    batch_size: int = 64,
    patch_size: int = 4,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """CIFAR-10 DataLoader — patchified.
    
    patch_size=4 → 32/4 = 8 → 8×8 = 64 patches per image
    input_dim = 4×4×3 = 48
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(root, train=True,  download=True, transform=train_tf)
    val_ds   = datasets.CIFAR10(root, train=False, download=True, transform=val_tf)

    collator = PatchCollator(patch_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collator, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collator, pin_memory=True,
    )

    return train_loader, val_loader
