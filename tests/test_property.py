"""Property-based tests (hypothesis): across random shapes the model must
produce finite outputs, finite losses, and finite gradients, and be
deterministic on CPU under a fixed seed. Catches NaN-on-edge-case bugs that
fixed-shape tests miss.
"""

from __future__ import annotations

import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from prism.config import ModalityConfig, PRISMConfig
from prism.model import PRISMForClassification

_SETTINGS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _model(num_layers, ssm_kind, in_dim, n_cls):
    cfg = PRISMConfig(
        hidden_dim=16,
        num_heads=2,
        num_layers=num_layers,
        delta_every=2,
        ssm_kind=ssm_kind,
        modalities=[ModalityConfig("m", in_dim, n_cls)],
    )
    return PRISMForClassification(cfg)


@_SETTINGS
@given(
    B=st.integers(1, 3),
    L=st.integers(1, 40),
    in_dim=st.integers(1, 16),
    n_cls=st.integers(2, 8),
    num_layers=st.integers(1, 4),
    ssm_kind=st.sampled_from(["ssd", "s4d_legacy"]),
)
def test_finite_forward_backward(B, L, in_dim, n_cls, num_layers, ssm_kind):
    torch.manual_seed(0)
    model = _model(num_layers, ssm_kind, in_dim, n_cls)
    x = torch.randn(B, L, in_dim)
    labels = torch.randint(0, n_cls, (B,))
    out = model(x, modality="m", labels=labels)
    assert out["logits"].shape == (B, n_cls)
    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["loss"]).all()
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads)


@_SETTINGS
@given(
    B=st.integers(1, 2),
    L=st.integers(1, 24),
    ssm_kind=st.sampled_from(["ssd", "s4d_legacy"]),
)
def test_cpu_determinism(B, L, ssm_kind):
    def run():
        torch.manual_seed(123)
        model = _model(2, ssm_kind, 8, 4)
        torch.manual_seed(7)
        x = torch.randn(B, L, 8)
        return model(x, modality="m")["logits"]

    torch.testing.assert_close(run(), run())
