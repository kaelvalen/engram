from __future__ import annotations

import numpy as np
import torch
from engram.config import ENGRAMConfig, ModalityConfig
from engram.model import ENGRAMForClassification
from scripts.infer_ecg import infer_signal


def test_infer_signal_uses_multilabel_scores(tmp_path):
    cfg = ENGRAMConfig(
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        scan_backend="reference",
        modalities=[
            ModalityConfig(
                name="ecg",
                input_dim=12,
                num_classes=5,
                multilabel=True,
                task="superdiag",
            )
        ],
    )
    model = ENGRAMForClassification(cfg).eval()
    with torch.no_grad():
        model.head.heads["ecg"].weight.zero_()
        model.head.heads["ecg"].bias.zero_()
    signal_path = tmp_path / "signal.npy"
    np.save(signal_path, np.random.default_rng(0).normal(size=(12, 12)).astype(np.float32))

    scores = infer_signal(model, str(signal_path), torch.device("cpu"))

    assert set(scores) == {"NORM", "MI", "STTC", "CD", "HYP"}
    assert all(0.0 <= value <= 1.0 for value in scores.values())
    assert np.isclose(sum(scores.values()), 2.5)
