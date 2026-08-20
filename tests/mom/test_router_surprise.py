"""Surprise-gated routing (router interface, Step 2 of repo unification).

The router can optionally consume a per-token `surprise: [B, T]` feature
(e.g. SABER's normalized SurpriseEstimator output). Default `surprise_scale=0`
must be byte-for-byte identical to the plain learned router; `> 0` must shift
logits and let gradient flow. Non-learned modes ignore surprise entirely.
"""

import torch
from mom.router import TokenRouter


def _router(K=3, k=1, D=16, **kw):
    return TokenRouter(hidden_dim=D, num_experts=K, top_k=k, **kw)


def _surprise(B=4, T=8, device="cpu"):
    return torch.linspace(0.0, 1.0, B * T, device=device).view(B, T)


def test_surprise_scale_zero_is_identical_regression():
    """surprise_scale=0 (default) => surprise arg must not change anything."""
    r = _router(K=3, k=1)
    h = torch.randn(4, 8, 16)
    s = _surprise(4, 8)
    base = r.forward(h)
    with_s = r.forward(h, surprise=s)
    assert torch.equal(base.logits, with_s.logits)
    assert torch.equal(base.indices, with_s.indices)
    assert torch.equal(base.gates, with_s.gates)
    assert torch.allclose(base.probs, with_s.probs)


def test_surprise_shifts_logits_and_route_when_scaled():
    """surprise_scale>0 in learned mode: logits shift by surprise*scale."""
    r = _router(K=3, k=1, surprise_scale=2.0)
    h = torch.randn(2, 5, 16)
    s = torch.zeros(2, 5)
    # zero surprise still uses linear path only, but nonzero must shift logits.
    s_nonzero = torch.linspace(0.0, 1.0, 10).view(2, 5)
    base = r.forward(h)
    w = r.forward(h, surprise=s_nonzero)
    expected = base.logits + 2.0 * s_nonzero.unsqueeze(-1)
    assert torch.allclose(w.logits, expected, atol=1e-6)
    assert not torch.allclose(base.logits, w.logits)
    # indices may change (not guaranteed) but shapes must agree.
    assert w.indices.shape == base.indices.shape


def test_gradient_flows_to_router_weight():
    """With surprise_scale>0, logits (and thus loss) depend on router weight."""
    r = _router(K=2, k=1, surprise_scale=1.0)
    h = torch.randn(3, 4, 16, requires_grad=False)
    s = torch.linspace(0.0, 1.0, 12).view(3, 4)
    out = r.forward(h, surprise=s)
    loss = out.logits.sum()
    loss.backward()
    assert r.weight.grad is not None
    assert r.weight.grad.abs().sum() > 0


def test_surprise_wrong_shape_raises():
    import pytest

    r = _router(K=3, k=1, surprise_scale=1.0)
    h = torch.randn(4, 8, 16)
    with pytest.raises(ValueError):
        r.forward(h, surprise=torch.randn(4, 7))  # T mismatch


def test_modes_ignore_surprise():
    """uniform/random modes don't read surprise: no shape crash, same output."""
    for mode in ("uniform", "random"):
        h = torch.randn(2, 6, 16)
        s = _surprise(2, 6)
        # Fresh router per call: the "random" mode consumes a seeded generator,
        # so a single instance advanced between calls would differ regardless.
        base = _router(K=3, k=2, mode=mode, surprise_scale=1.0, seed=0).forward(h)
        with_s = _router(K=3, k=2, mode=mode, surprise_scale=1.0, seed=0).forward(h, surprise=s)
        assert torch.equal(base.gates, with_s.gates)
        assert torch.equal(base.indices, with_s.indices)
