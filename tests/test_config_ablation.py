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
