from __future__ import annotations

import pytest
import torch
from prism.config import ModalityConfig, PRISMConfig
from prism.model import PRISMForClassification
from prism.training.checkpoint import load_checkpoint, save_checkpoint


def _tiny_cfg() -> PRISMConfig:
    return PRISMConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=4,
        delta_every=2,
        modalities=[ModalityConfig(name="ecg", input_dim=12, num_classes=3)],
    )


def test_load_checkpoint_raises_on_missing_model_state(tmp_path):
    bad = {"cfg": {"hidden_dim": 32}}
    path = tmp_path / "bad.pt"
    torch.save(bad, path)
    with pytest.raises(ValueError, match="model_state"):
        load_checkpoint(path)


def test_load_checkpoint_raises_on_missing_cfg(tmp_path):
    bad = {"model_state": {}}
    path = tmp_path / "bad.pt"
    torch.save(bad, path)
    with pytest.raises(ValueError, match="cfg"):
        load_checkpoint(path)


def test_load_checkpoint_roundtrip(tmp_path):
    cfg = _tiny_cfg()
    model = PRISMForClassification(cfg)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, epoch=1, model_state=model.state_dict(), cfg=cfg)
    ckpt = load_checkpoint(path)
    assert "model_state" in ckpt
    assert "cfg" in ckpt
    assert ckpt["epoch"] == 1
