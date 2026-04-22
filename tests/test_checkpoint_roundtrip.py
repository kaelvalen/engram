from __future__ import annotations

from pathlib import Path

import torch
from prism.config import ModalityConfig, PRISMConfig
from prism.integrations.huggingface import load_pretrained_folder, save_pretrained_folder
from prism.model import PRISMForClassification


def test_save_load_hf_folder(tmp_path: Path):
    cfg = PRISMConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=4,
        delta_every=2,
        modalities=[ModalityConfig(name="ecg", input_dim=12, num_classes=3)],
    )
    m = PRISMForClassification(cfg)
    x = torch.randn(2, 8, 12)
    y = torch.randint(0, 3, (2,))
    before = m(x, modality="ecg", labels=y)["logits"].detach()

    d = tmp_path / "hf_ckpt"
    save_pretrained_folder(m, d)
    m2 = load_pretrained_folder(d, map_location="cpu")
    after = m2(x, modality="ecg", labels=y)["logits"].detach()
    assert torch.allclose(before, after)
