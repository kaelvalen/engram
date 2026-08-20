from __future__ import annotations

from pathlib import Path

import torch
from engram.config import ENGRAMConfig, ModalityConfig
from engram.integrations.huggingface import load_pretrained_folder, save_pretrained_folder
from engram.model import ENGRAMForClassification


def test_save_load_hf_folder(tmp_path: Path):
    cfg = ENGRAMConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=4,
        delta_every=2,
        modalities=[ModalityConfig(name="ecg", input_dim=12, num_classes=3)],
    )
    m = ENGRAMForClassification(cfg)
    x = torch.randn(2, 8, 12)
    y = torch.randint(0, 3, (2,))
    before = m(x, modality="ecg", labels=y)["logits"].detach()

    d = tmp_path / "hf_ckpt"
    save_pretrained_folder(m, d)
    m2 = load_pretrained_folder(d, map_location="cpu")
    after = m2(x, modality="ecg", labels=y)["logits"].detach()
    assert torch.allclose(before, after)
