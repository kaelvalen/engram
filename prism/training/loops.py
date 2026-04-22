from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from prism.model import PRISMForClassification


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def train_epoch(
    model: PRISMForClassification,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    modality: str,
    *,
    max_grad_norm: float = 1.0,
    loss_log_fn: Callable[[str, float, int], None] | None = None,
) -> tuple[float, float]:
    model.train(True)
    total_loss, total_acc, n = 0.0, 0.0, 0

    for step, (x, labels) in enumerate(loader):
        x, labels = x.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(x, modality=modality, labels=labels)
        out["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        B = x.size(0)
        total_loss += out["loss"].item() * B
        total_acc += accuracy(out["logits"], labels) * B
        n += B
        if loss_log_fn is not None:
            loss_log_fn("train/loss_step", out["loss"].item(), step)

    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate_epoch(
    model: PRISMForClassification,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    modality: str,
) -> tuple[float, float]:
    model.train(False)
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, labels in loader:
        x, labels = x.to(device), labels.to(device)
        out = model(x, modality=modality, labels=labels)

        B = x.size(0)
        total_loss += out["loss"].item() * B
        total_acc += accuracy(out["logits"], labels) * B
        n += B

    return total_loss / n, total_acc / n


def cycle_loader(loader: torch.utils.data.DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch
