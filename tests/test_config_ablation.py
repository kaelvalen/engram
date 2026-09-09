from __future__ import annotations

from engram.config import ENGRAMConfig, ModalityConfig


def test_force_s4_all_layers():
    cfg = ENGRAMConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=6,
        delta_every=2,
        force_block_type="s4",
        modalities=[ModalityConfig(name="ecg", input_dim=12, num_classes=5)],
    )
    pat = cfg.layer_pattern()
    assert pat == ["s4"] * 6


def test_force_delta_all_layers():
    cfg = ENGRAMConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=4,
        delta_every=2,
        force_block_type="delta",
        modalities=[ModalityConfig(name="ecg", input_dim=12, num_classes=5)],
    )
    assert cfg.layer_pattern() == ["delta"] * 4


def test_hybrid_respects_delta_every():
    cfg = ENGRAMConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=4,
        delta_every=2,
        modalities=[ModalityConfig(name="ecg", input_dim=12, num_classes=5)],
    )
    assert cfg.layer_pattern() == ["s4", "delta", "s4", "delta"]


def test_delta_out_gate_ablation_forward():
    import torch
    from engram.model import ENGRAMForClassification

    cfg = ENGRAMConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        force_block_type="delta",
        delta_out_gate=False,
        modalities=[ModalityConfig(name="ecg", input_dim=12, num_classes=5)],
    )
    model = ENGRAMForClassification(cfg)
    x = torch.randn(2, 16, 12)
    y = torch.randint(0, 5, (2,))
    out = model(x, modality="ecg", labels=y)
    assert out["logits"].shape == (2, 5)
    out["loss"].backward()
    assert any(p.grad is not None for p in model.parameters())


def test_delta_memoryless_ablation_forward():
    import torch
    from engram.model import ENGRAMForClassification

    cfg = ENGRAMConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        force_block_type="delta",
        delta_memoryless=True,
        modalities=[ModalityConfig(name="ecg", input_dim=12, num_classes=5)],
    )
    model = ENGRAMForClassification(cfg)
    x = torch.randn(2, 16, 12)
    y = torch.randint(0, 5, (2,))
    out = model(x, modality="ecg", labels=y)
    assert out["logits"].shape == (2, 5)
    out["loss"].backward()
    assert any(p.grad is not None for p in model.parameters())
