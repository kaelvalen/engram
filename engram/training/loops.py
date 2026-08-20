from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from engram.model import ENGRAMForClassification


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.dim() not in (1, 2):
        raise ValueError(f"accuracy expects labels of dim 1 or 2, got {labels.dim()}")
    if labels.numel() == 0:
        raise ValueError("accuracy received empty labels")
    if labels.dim() == 2:
        # multi-label: label-wise accuracy at a 0.5 probability threshold (a
        # training-progress proxy; the reported metric is macro AUROC via
        # evaluate_multilabel_auc).
        preds = (logits.sigmoid() > 0.5).float()
        return (preds == labels).float().mean().item()
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def _autocast(device: torch.device, amp_dtype: torch.dtype | None):
    if amp_dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=amp_dtype)


def train_epoch(
    model: ENGRAMForClassification,
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

    if n == 0:
        raise ValueError("train_epoch received an empty loader")
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate_epoch(
    model: ENGRAMForClassification,
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

    if n == 0:
        raise ValueError("evaluate_epoch received an empty loader")
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate_macro_auc(
    model: ENGRAMForClassification,
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
    from engram.training.metrics import roc_auc_ovr_macro

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


@torch.no_grad()
def evaluate_multilabel_auc(
    model: ENGRAMForClassification,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    modality: str,
    *,
    amp_dtype: torch.dtype | None = None,
) -> float:
    """Macro AUROC for multi-label targets ([N, C] multi-hot) — the PTB-XL
    all/diag/super-diag/form/rhythm metric. Accumulates sigmoid scores + targets
    across the loader and computes one macro AUROC.
    """
    from engram.training.metrics import multilabel_auroc_macro

    model.train(False)
    all_scores, all_targets = [], []
    for x, labels in loader:
        x = x.to(device)
        with _autocast(device, amp_dtype):
            out = model(x, modality=modality)
        all_scores.append(torch.sigmoid(out["logits"].float()).cpu())
        all_targets.append(labels.cpu())
    return multilabel_auroc_macro(torch.cat(all_scores), torch.cat(all_targets))


def cycle_loader(loader: torch.utils.data.DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch
