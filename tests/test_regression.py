"""Seed-locked regression guards. Protect against silent numerical drift when
refactoring: a fully-seeded forward must reproduce a recorded loss, and a short
training run must drive that loss down. Golden values recorded on CPU / torch
2.x; loosen rtol only with a deliberate, reviewed reason.
"""

from __future__ import annotations

import pytest
import torch
from engram.config import ENGRAMConfig, ModalityConfig
from engram.model import ENGRAMForClassification

# Recorded golden initial-forward losses (seed 1234 model, seed 99 data).
GOLDEN_LOSS0 = {"ssd": 1.606000, "s4d_legacy": 1.817113}


def _build(ssm_kind):
    torch.manual_seed(1234)
    cfg = ENGRAMConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=4,
        delta_every=2,
        ssm_kind=ssm_kind,
        scan_backend="reference",
        modalities=[ModalityConfig("ecg", 12, 5)],
    )
    return ENGRAMForClassification(cfg)


def _data():
    torch.manual_seed(99)
    return torch.randn(8, 32, 12), torch.randint(0, 5, (8,))


@pytest.mark.parametrize("ssm_kind", ["ssd", "s4d_legacy"])
def test_initial_forward_loss_matches_golden(ssm_kind):
    model = _build(ssm_kind).eval()
    x, y = _data()
    with torch.no_grad():
        loss0 = model(x, "ecg", labels=y)["loss"].item()
    assert loss0 == pytest.approx(GOLDEN_LOSS0[ssm_kind], rel=1e-3)


@pytest.mark.parametrize("ssm_kind", ["ssd", "s4d_legacy"])
def test_short_training_reduces_loss(ssm_kind):
    model = _build(ssm_kind)
    x, y = _data()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(30):
        opt.zero_grad()
        model(x, "ecg", labels=y)["loss"].backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        lossN = model(x, "ecg", labels=y)["loss"].item()
    # 30 steps must overfit this tiny batch well below half the initial loss.
    assert lossN < 0.5 * GOLDEN_LOSS0[ssm_kind]
