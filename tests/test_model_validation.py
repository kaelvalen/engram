from __future__ import annotations

import pytest
import torch
from prism.config import ModalityConfig, PRISMConfig
from prism.model import PRISMForClassification


def _tiny_cfg() -> PRISMConfig:
    return PRISMConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=4,
        delta_every=2,
        modalities=[
            ModalityConfig(name="ecg", input_dim=12, num_classes=5),
            ModalityConfig(name="image", input_dim=48, num_classes=10),
        ],
    )


def test_forward_raises_on_wrong_input_dim():
    model = PRISMForClassification(_tiny_cfg())
    x = torch.randn(2, 32, 99)  # wrong: expected 12 for "ecg"
    with pytest.raises(ValueError, match="Input last dim 99"):
        model(x, modality="ecg")


def test_forward_raises_on_unknown_modality():
    model = PRISMForClassification(_tiny_cfg())
    with pytest.raises(KeyError, match="unknown"):
        model(torch.randn(1, 8, 12), modality="unknown")


def test_forward_batch_size_one():
    model = PRISMForClassification(_tiny_cfg())
    x = torch.randn(1, 32, 12)
    out = model(x, modality="ecg")
    assert out["logits"].shape == (1, 5)


def test_forward_sequence_length_one():
    model = PRISMForClassification(_tiny_cfg())
    x = torch.randn(2, 1, 12)
    out = model(x, modality="ecg")
    assert out["logits"].shape == (2, 5)


def test_nan_input_propagates():
    model = PRISMForClassification(_tiny_cfg())
    x = torch.full((2, 8, 12), float("nan"))
    out = model(x, modality="ecg")
    assert torch.isnan(out["logits"]).any(), "NaN input should propagate to output"
