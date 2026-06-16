"""Macro AUROC metric correctness (dependency-free, rank-based)."""

from __future__ import annotations

import torch
from prism.training.metrics import binary_auroc, multilabel_auroc_macro, roc_auc_ovr_macro


def test_perfect_separation():
    assert binary_auroc(torch.tensor([0.1, 0.2, 0.9, 0.8]), torch.tensor([0, 0, 1, 1])) == 1.0


def test_perfectly_wrong():
    assert binary_auroc(torch.tensor([0.9, 0.8, 0.1, 0.2]), torch.tensor([0, 0, 1, 1])) == 0.0


def test_all_ties_is_half():
    assert binary_auroc(torch.tensor([0.5, 0.5, 0.5, 0.5]), torch.tensor([0, 1, 0, 1])) == 0.5


def test_degenerate_returns_none():
    assert binary_auroc(torch.randn(5), torch.zeros(5)) is None
    assert binary_auroc(torch.randn(5), torch.ones(5)) is None


def test_invariant_to_monotonic_score_shift():
    torch.manual_seed(0)
    s = torch.randn(100)
    y = (torch.rand(100) > 0.5).int()
    base = binary_auroc(s, y)
    shifted = binary_auroc(s * 3.0 + 7.0, y)  # monotonic transform → same AUC
    assert abs(base - shifted) < 1e-6


def test_ovr_macro_perfect():
    # logits that perfectly rank each class highest for its own samples
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0], [4.0, 0.0, 0.0]])
    labels = torch.tensor([0, 1, 2, 0])
    assert roc_auc_ovr_macro(logits, labels, 3) == 1.0


def test_multilabel_macro_range():
    torch.manual_seed(1)
    scores = torch.randn(50, 4)
    targets = (torch.rand(50, 4) > 0.5).int()
    auc = multilabel_auroc_macro(scores, targets)
    assert 0.0 <= auc <= 1.0
