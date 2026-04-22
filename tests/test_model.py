from __future__ import annotations

import pytest
import torch
from prism.config import ModalityConfig, PRISMConfig
from prism.model import PRISMForClassification


def tiny_cfg(**kwargs) -> PRISMConfig:
    base = dict(
        hidden_dim=32,
        num_heads=4,
        num_layers=4,
        delta_every=2,
        modalities=[
            ModalityConfig(name="ecg", input_dim=12, num_classes=5),
            ModalityConfig(name="image", input_dim=48, num_classes=10),
        ],
    )
    base.update(kwargs)
    return PRISMConfig(**base)


def test_forward_ecg_and_image():
    cfg = tiny_cfg()
    model = PRISMForClassification(cfg)
    b = 2
    ecg = torch.randn(b, 32, 12)
    y = torch.randint(0, 5, (b,))
    out = model(ecg, modality="ecg", labels=y)
    assert out["logits"].shape == (b, 5)
    assert out["loss"].shape == ()

    img = torch.randn(b, 16, 48)
    yi = torch.randint(0, 10, (b,))
    out_i = model(img, modality="image", labels=yi)
    assert out_i["logits"].shape == (b, 10)


def test_unknown_modality_raises():
    cfg = tiny_cfg()
    model = PRISMForClassification(cfg)
    with pytest.raises(KeyError):
        model(torch.randn(1, 8, 12), modality="unknown", labels=torch.zeros(1, dtype=torch.long))


def test_backward_step():
    cfg = tiny_cfg()
    model = PRISMForClassification(cfg)
    x = torch.randn(2, 16, 12, requires_grad=False)
    y = torch.randint(0, 5, (2,))
    out = model(x, modality="ecg", labels=y)
    out["loss"].backward()
    assert any(p.grad is not None for p in model.parameters())
