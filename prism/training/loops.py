from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from prism.model import PRISMForClassification


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def _autocast(device: torch.device, amp_dtype: torch.dtype | None):
    if amp_dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=amp_dtype)


def train_epoch(
    model: PRISMForClassification,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    modality: str,
    *,
    max_grad_norm: float = 1.0,
    loss_log_fn: Callable[[str, float, int], None] | None = None,
    amp_dtype: torch.dtype | None = None,
) -> tuple[float, float]:
    model.train(True)
    total_loss, total_acc, n = 0.0, 0.0, 0

    for step, (x, labels) in enumerate(loader):
        x, labels = x.to(device), labels.to(device)
        optimizer.zero_grad()
        with _autocast(device, amp_dtype):
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
    *,
    amp_dtype: torch.dtype | None = None,
) -> tuple[float, float]:
    model.train(False)
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, labels in loader:
        x, labels = x.to(device), labels.to(device)
        with _autocast(device, amp_dtype):
            out = model(x, modality=modality, labels=labels)

        B = x.size(0)
        total_loss += out["loss"].item() * B
        total_acc += accuracy(out["logits"], labels) * B
        n += B

    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate_macro_auc(
    model: PRISMForClassification,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    modality: str,
    num_classes: int,
    *,
    amp_dtype: torch.dtype | None = None,
) -> float:
    """Macro one-vs-rest AUROC over the whole loader — the PTB-XL metric.

    Accumulates logits/labels across batches (datasets are small) and computes
    a single macro AUROC, matching the Strodthoff et al. evaluation protocol.
    """
    from prism.training.metrics import roc_auc_ovr_macro

    model.train(False)
    all_logits, all_labels = [], []
    for x, labels in loader:
        x = x.to(device)
        with _autocast(device, amp_dtype):
            out = model(x, modality=modality)
        all_logits.append(out["logits"].float().cpu())
        all_labels.append(labels.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return roc_auc_ovr_macro(logits, labels, num_classes)


def cycle_loader(loader: torch.utils.data.DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch
