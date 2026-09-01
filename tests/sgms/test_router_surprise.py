"""Surprise-gated routing (router interface, post design-review rework).

The surprise feature must be **expert-dependent** to be able to affect routing:
a scalar broadcast over all experts is softmax/argmax shift-invariant and hence
a no-op for top_k=1. The router therefore folds `surprise: [B, T]` through a
per-expert `surprise_weight` (shape [K]) scaled by `surprise_scale`.

Defaults are inert (surprise_scale == 0, or all-zero surprise_weight) so the
plain router and every existing test are unchanged. Non-learned modes ignore
surprise entirely.
"""

import torch
from sgms.router import TokenRouter


def _router(K=3, k=1, D=16, **kw):
    return TokenRouter(hidden_dim=D, num_experts=K, top_k=k, **kw)


def _surprise(B=4, T=8, device="cpu"):
    return torch.linspace(0.0, 1.0, B * T, device=device).view(B, T)


def test_scale_zero_is_identical_regression():
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


def test_zero_surprise_weight_is_inert():
    """scale>0 but surprise_weight stays zero => fully inert (backward-compatible)."""
    r = _router(K=3, k=1, surprise_scale=5.0)
    h = torch.randn(2, 6, 16)
    s = _surprise(2, 6)
    base = r.forward(h)
    with_s = r.forward(h, surprise=s)
    assert torch.equal(base.logits, with_s.logits)
    assert torch.equal(base.indices, with_s.indices)


def test_per_expert_weight_shifts_logits_and_can_flip_decision():
    """Nonzero per-expert surprise_weight changes logits and can change routing."""
    r = _router(K=3, k=1, surprise_scale=1.0)
    with torch.no_grad():
        r.surprise_weight.copy_(torch.tensor([1.0, -1.0, -0.5]))
    h = torch.randn(2, 6, 16)
    s = _surprise(2, 6)
    base = r.forward(h)
    w = r.forward(h, surprise=s)
    for t in range(6):
        # expected logit shift is scale * w_e * surprise
        for e in range(3):
            shift = 1.0 * r.surprise_weight[e].item() * s[0, t].item()
            assert torch.allclose(w.logits[0, t, e], base.logits[0, t, e] + shift, atol=1e-6)
    # a strong, expert-discriminating surprise must be able to flip routing
    # for at least one token in the batch
    delivered = 0
    for b in range(2):
        for t in range(6):
            if base.indices[b, t].item() != w.indices[b, t].item():
                delivered += 1
    assert delivered > 0, "per-expert surprise never changed a routing decision"


def test_gradient_flows_to_router_and_surprise_weight():
    """Task loss through logits reaches both router.weight and surprise_weight."""
    r = _router(K=2, k=1, surprise_scale=1.0)
    h = torch.randn(3, 4, 16)
    s = torch.linspace(0.0, 1.0, 12).view(3, 4)
    out = r.forward(h, surprise=s)
    out.logits.sum().backward()
    assert r.weight.grad is not None and r.weight.grad.abs().sum() > 0
    # surprise_weight must receive a gradient through its own contribution
    assert r.surprise_weight.grad is not None
    assert r.surprise_weight.grad.abs().sum() > 0


def test_surprise_wrong_shape_raises():
    import pytest

    r = _router(K=3, k=1, surprise_scale=1.0)
    h = torch.randn(4, 8, 16)
    with pytest.raises(ValueError):
        r.forward(h, surprise=torch.randn(4, 7))  # T mismatch


def test_modes_ignore_surprise():
    """uniform/random modes ignore surprise: no shape crash, same output."""
    for mode in ("uniform", "random"):
        h = torch.randn(2, 6, 16)
        s = _surprise(2, 6)
        # Fresh router per call: "random" consumes a seeded generator, so a
        # shared instance advanced between calls would differ regardless.
        base = _router(K=3, k=2, mode=mode, surprise_scale=1.0, seed=0).forward(h)
        with_s = _router(K=3, k=2, mode=mode, surprise_scale=1.0, seed=0).forward(h, surprise=s)
        assert torch.equal(base.gates, with_s.gates)
        assert torch.equal(base.indices, with_s.indices)
