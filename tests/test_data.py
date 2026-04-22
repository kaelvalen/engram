from __future__ import annotations

import pytest
import torch
from prism.data.audio import SyntheticMelPatchDataset
from prism.data.ecg import _check_ecg_failure_rate
from prism.data.image import patchify

# ── ECG failure gate ──────────────────────────────────────────────────────────

def test_ecg_failure_gate_raises_on_high_rate():
    """50% failure rate (5/10) exceeds threshold."""
    with pytest.raises(RuntimeError, match="failure rate"):
        _check_ecg_failure_rate(list(range(5)), 10)


def test_ecg_failure_gate_raises_at_boundary():
    """Exactly 11% failure (2/18 = 11.1%) should raise."""
    with pytest.raises(RuntimeError, match="failure rate"):
        _check_ecg_failure_rate(list(range(2)), 18)


def test_ecg_failure_gate_passes_under_threshold():
    """5% failure (1/20) is below 10% threshold — should not raise."""
    _check_ecg_failure_rate([1], 20)  # 5% — OK


def test_ecg_failure_gate_passes_on_zero_total():
    """Empty dataset should not raise (no divide-by-zero)."""
    _check_ecg_failure_rate([], 0)


# ── Image ─────────────────────────────────────────────────────────────────────

def test_patchify_raises_on_indivisible_size():
    x = torch.randn(2, 3, 32, 32)
    with pytest.raises(AssertionError):
        patchify(x, patch_size=5)  # 32 % 5 != 0


def test_patchify_correct_output_shape():
    x = torch.randn(4, 3, 32, 32)
    out = patchify(x, patch_size=4)
    assert out.shape == (4, 64, 48)  # 8×8 patches, 4×4×3=48 dim


# ── Audio ─────────────────────────────────────────────────────────────────────

def test_audio_synthetic_is_deterministic():
    ds1 = SyntheticMelPatchDataset(length=8, num_frames=32, mel_bins=16, num_classes=4, seed=42)
    ds2 = SyntheticMelPatchDataset(length=8, num_frames=32, mel_bins=16, num_classes=4, seed=42)
    x1, y1 = ds1[0]
    x2, y2 = ds2[0]
    assert torch.allclose(x1, x2)
    assert y1 == y2


def test_audio_different_seeds_differ():
    ds1 = SyntheticMelPatchDataset(length=8, num_frames=32, mel_bins=16, num_classes=4, seed=1)
    ds2 = SyntheticMelPatchDataset(length=8, num_frames=32, mel_bins=16, num_classes=4, seed=2)
    x1, _ = ds1[0]
    x2, _ = ds2[0]
    assert not torch.allclose(x1, x2)
