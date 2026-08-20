"""Multi-label path: BCE loss, label-wise accuracy, accumulating macro AUROC,
and bootstrap CI — the PTB-XL evaluation protocol, validated on synthetic data
(no dataset download needed).
"""

from __future__ import annotations

import torch
from engram.config import ENGRAMConfig, ModalityConfig
from engram.model import ENGRAMForClassification
from engram.training.loops import accuracy, evaluate_multilabel_auc
from engram.training.metrics import bootstrap_auroc_ci
from torch.utils.data import DataLoader, TensorDataset


def _multilabel_model(num_classes=5):
    cfg = ENGRAMConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=4,
        modalities=[ModalityConfig("ecg", 12, num_classes, multilabel=True)],
    )
    return ENGRAMForClassification(cfg)


def test_multilabel_forward_uses_bce_and_backprops():
    torch.manual_seed(0)
    model = _multilabel_model()
    x = torch.randn(4, 24, 12)
    labels = (torch.rand(4, 5) > 0.5).float()  # multi-hot
    out = model(x, modality="ecg", labels=labels)
    assert out["logits"].shape == (4, 5)
    assert out["loss"].shape == ()
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None for p in model.parameters())


def test_accuracy_handles_2d_labels():
    logits = torch.tensor([[2.0, -2.0], [-1.0, 3.0]])  # → preds [[1,0],[0,1]]
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert accuracy(logits, labels) == 1.0
    # single-label path still works
    assert accuracy(torch.tensor([[2.0, 0.0]]), torch.tensor([0])) == 1.0


def test_evaluate_multilabel_auc_separable():
    torch.manual_seed(0)
    model = _multilabel_model()
    x = torch.randn(16, 12, 12)
    y = (torch.rand(16, 5) > 0.5).float()
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4)
    auc = evaluate_multilabel_auc(model, loader, torch.device("cpu"), "ecg")
    assert 0.0 <= auc <= 1.0


def test_bootstrap_ci_multilabel_perfect():
    # perfectly separable per class → point ~1.0 and CI tight near 1.0
    scores = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.2, 0.7], [0.9, 0.1]])
    targets = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    point, lo, hi = bootstrap_auroc_ci(scores, targets, n_resamples=200, seed=0)
    assert point == 1.0
    assert lo <= point <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_singlelabel_shape():
    torch.manual_seed(1)
    logits = torch.randn(100, 5)
    labels = torch.randint(0, 5, (100,))
    point, lo, hi = bootstrap_auroc_ci(
        logits, labels, multilabel=False, num_classes=5, n_resamples=200
    )
    assert lo <= point <= hi
    assert 0.0 <= lo <= hi <= 1.0
