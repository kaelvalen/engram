from __future__ import annotations

import torch
from engram.baselines import (
    AudioCNNClassifier,
    CompactConvNet2D,
    ResNet1DClassifier,
    TransformerSequenceClassifier,
)


def test_resnet1d_standard_and_wide():
    # Standard: ~127k params
    m_std = ResNet1DClassifier(in_channels=12, num_classes=5)
    params_std = sum(p.numel() for p in m_std.parameters())
    assert 120_000 <= params_std <= 135_000

    # Wide: ~254k params (parameter-matched to engram_hybrid_ecg)
    m_wide = ResNet1DClassifier(in_channels=12, num_classes=5, base_channels=92)
    params_wide = sum(p.numel() for p in m_wide.parameters())
    assert 250_000 <= params_wide <= 260_000

    # Forward & backward with multilabel targets
    x = torch.randn(2, 1000, 12)
    labels = torch.zeros(2, 5)
    labels[0, 1] = 1.0
    labels[1, 3] = 1.0
    out = m_wide(x, labels=labels)
    assert out["logits"].shape == (2, 5)
    assert out["loss"].shape == ()
    out["loss"].backward()
    assert any(p.grad is not None for p in m_wide.parameters())


def test_compact_convnet2d():
    # Matched: ~269k params (channels 32, 48, 64, 96)
    m = CompactConvNet2D(in_channels=3, num_classes=10, channels=(32, 48, 64, 96))
    params = sum(p.numel() for p in m.parameters())
    assert 250_000 <= params <= 280_000

    # Patchified input [B, 64, 48]
    x_patch = torch.randn(2, 64, 48)
    labels = torch.randint(0, 10, (2,))
    out_patch = m(x_patch, labels=labels)
    assert out_patch["logits"].shape == (2, 10)
    out_patch["loss"].backward()
    assert any(p.grad is not None for p in m.parameters())

    # Raw 2D input [B, 3, 32, 32]
    x_raw = torch.randn(2, 3, 32, 32)
    out_raw = m(x_raw)
    assert out_raw["logits"].shape == (2, 10)


def test_audiocnn_standard_and_matched():
    # Matched: base_channels=44 gives ~250k params
    m = AudioCNNClassifier(input_dim=64, num_classes=10, base_channels=44)
    params = sum(p.numel() for p in m.parameters())
    assert 240_000 <= params <= 270_000

    x = torch.randn(2, 32, 64)
    labels = torch.randint(0, 10, (2,))
    out = m(x, labels=labels)
    assert out["logits"].shape == (2, 10)
    out["loss"].backward()
    assert any(p.grad is not None for p in m.parameters())
