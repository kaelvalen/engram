"""TokenRouter tests (spec §3.3): top-k validity, determinism, mask algebra.

Router math:
    z_t = W_r h_t (+ b)           W_r ∈ R^{K×D}, bias-free by default
    S_t = indices of k largest z_t
    g_{t,e} = softmax(z_t)_e over e ∈ S_t, renormalised; 0 otherwise

Modes: ``learned`` (default), ``uniform`` (B4: g = 1/K frozen),
``random`` (B5: seeded random per-token assignment).
"""

from __future__ import annotations

import torch
from sgms.masking import topk_mask
from sgms.router import RoutingOutput, TokenRouter


def _router(K=3, k=1, D=16, **kw):
    torch.manual_seed(0)
    return TokenRouter(hidden_dim=D, num_experts=K, top_k=k, **kw)


# ---------------------------------------------------------------------------
# top-k validity / simplex constraints
# ---------------------------------------------------------------------------


def test_k1_exactly_one_expert_per_token():
    r = _router(K=3, k=1)
    out = r(torch.randn(2, 7, 16))
    assert (out.mask.sum(-1) == 1).all()
    assert ((out.gates > 0).sum(-1) == 1).all()
    torch.testing.assert_close(out.gates.sum(-1), torch.ones(2, 7))
    # k=1 renormalised gate is exactly 1 on the selected expert
    torch.testing.assert_close(out.gates.max(-1).values, torch.ones(2, 7))


def test_k2_two_experts_gates_on_simplex():
    r = _router(K=4, k=2)
    out = r(torch.randn(2, 7, 16))
    assert (out.mask.sum(-1) == 2).all()
    torch.testing.assert_close(out.gates.sum(-1), torch.ones(2, 7))
    assert (out.gates >= 0).all()
    # distinct experts per token
    assert (out.indices[..., 0] != out.indices[..., 1]).all()


def test_gates_zero_outside_topk():
    r = _router(K=4, k=2)
    out = r(torch.randn(2, 7, 16))
    hard = torch.zeros_like(out.gates).scatter(-1, out.indices, 1.0)
    assert torch.equal(hard, out.mask)
    assert torch.equal(out.gates * (1 - hard), torch.zeros_like(out.gates))


def test_indices_match_argmax():
    r = _router(K=3, k=1)
    h = torch.randn(2, 7, 16)
    out = r(h)
    z = h @ r.weight.T
    assert torch.equal(out.indices[..., 0], z.argmax(-1))


def test_probs_are_valid_softmax():
    r = _router(K=3, k=1)
    out = r(torch.randn(2, 7, 16))
    torch.testing.assert_close(out.probs.sum(-1), torch.ones(2, 7))
    assert out.logits is not None and out.logits.shape == (2, 7, 3)


def test_invalid_top_k_raises():
    with __import__("pytest").raises(ValueError):
        _router(K=2, k=3)
    with __import__("pytest").raises(ValueError):
        _router(K=2, k=0)


def test_invalid_mode_raises():
    with __import__("pytest").raises(ValueError):
        _router(mode="bogus")


# ---------------------------------------------------------------------------
# init / determinism
# ---------------------------------------------------------------------------


def test_near_uniform_initialisation():
    r = _router(K=4, k=1, init_std=0.01)
    out = r(torch.randn(4, 32, 16))
    # N(0, 0.01²) weights ⇒ near-uniform probabilities
    torch.testing.assert_close(out.probs, torch.full_like(out.probs, 0.25), rtol=0.35, atol=0.05)


def test_determinism_same_seed():
    h = torch.randn(2, 16, 16)
    a = _router().forward(h)
    b = _router().forward(h)
    assert torch.equal(a.indices, b.indices)
    assert torch.equal(a.gates, b.gates)


def test_bias_option_shifts_routing():
    r = _router(K=2, k=1, bias=True)
    with torch.no_grad():
        r.weight.zero_()
        r.router_bias.copy_(torch.tensor([0.0, 10.0]))
    out = r(torch.randn(2, 8, 16))
    assert (out.indices[..., 0] == 1).all()


def test_bias_free_by_default():
    r = _router()
    assert r.router_bias is None


# ---------------------------------------------------------------------------
# baseline modes (B4 / B5)
# ---------------------------------------------------------------------------


def test_uniform_mode_frozen_mixture():
    r = _router(K=3, k=1, mode="uniform")
    out = r(torch.randn(2, 5, 16))
    torch.testing.assert_close(out.gates, torch.full_like(out.gates, 1 / 3))
    assert (out.mask == 1).all()
    assert out.logits is None
    assert not r.weight.requires_grad


def test_random_mode_seeded_reproducible():
    h = torch.randn(2, 9, 16)
    a = _router(K=3, k=2, mode="random", seed=7).forward(h)
    b = _router(K=3, k=2, mode="random", seed=7).forward(h)
    c = _router(K=3, k=2, mode="random", seed=8).forward(h)
    assert torch.equal(a.indices, b.indices)
    assert not torch.equal(a.indices, c.indices)
    # k distinct experts, uniform gates over the selection
    assert (a.indices[..., 0] != a.indices[..., 1]).all()
    torch.testing.assert_close(a.gates.sum(-1), torch.ones(2, 9))
    torch.testing.assert_close(a.gates.max(-1).values, torch.full((2, 9), 0.5))


def test_random_mode_independent_of_input_values():
    r1 = _router(K=3, k=1, mode="random", seed=11)
    r2 = _router(K=3, k=1, mode="random", seed=11)
    a = r1(torch.zeros(1, 6, 16))
    b = r2(torch.randn(1, 6, 16))
    assert torch.equal(a.indices, b.indices)


# ---------------------------------------------------------------------------
# straight-through estimator (R4 fallback)
# ---------------------------------------------------------------------------


def test_k1_gate_path_has_no_gradient_by_default():
    r = _router(K=2, k=1)
    h = torch.randn(1, 4, 16)
    out = r(h)
    loss = (out.gates * torch.randn_like(out.gates)).sum()
    g = torch.autograd.grad(loss, r.weight, retain_graph=True, allow_unused=True)[0]
    assert g is None or torch.equal(g, torch.zeros_like(g))


def test_straight_through_restores_gate_gradient():
    r = _router(K=2, k=1, straight_through=True)
    h = torch.randn(1, 4, 16)
    out = r(h)
    loss = (out.gates * torch.randn_like(out.gates)).sum()
    g = torch.autograd.grad(loss, r.weight, allow_unused=True)[0]
    assert g is not None and g.abs().sum() > 0


def test_straight_through_preserves_forward_values():
    kw = dict(K=2, k=2, D=16)
    a = _router(**kw)
    b = _router(**kw, straight_through=True)
    b.load_state_dict(a.state_dict())
    h = torch.randn(1, 4, 16)
    torch.testing.assert_close(a(h).gates, b(h).gates)


# ---------------------------------------------------------------------------
# knockout exclusion (analysis §7.3)
# ---------------------------------------------------------------------------


def test_excluded_expert_never_selected():
    r = _router(K=3, k=1)
    out = r(torch.randn(2, 9, 16), exclude={0})
    assert (out.indices != 0).all()
    torch.testing.assert_close(out.gates[..., 0], torch.zeros(2, 9))
    torch.testing.assert_close(out.gates.sum(-1), torch.ones(2, 9))


def test_exclusion_renormalises_k2():
    r = _router(K=4, k=2)
    out = r(torch.randn(2, 9, 16), exclude={1, 3})
    assert (out.mask[..., [1, 3]] == 0).all()
    assert (out.mask.sum(-1) == 2).all()


# ---------------------------------------------------------------------------
# mask algebra (sgms.masking)
# ---------------------------------------------------------------------------


def test_topk_mask_scatter():
    idx = torch.tensor([[[0], [2]], [[1], [0]]])
    m = topk_mask(idx, 3)
    assert m.shape == (2, 2, 3)
    assert torch.equal(m[0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(m[0, 1], torch.tensor([0.0, 0.0, 1.0]))
    assert (m.sum(-1) == 1).all()


def test_mask_idempotent():
    idx = torch.randint(0, 3, (2, 8, 1))
    m = topk_mask(idx, 3)
    assert torch.equal(m * m, m)


def test_mask_dtype_float():
    m = topk_mask(torch.zeros(1, 1, 1, dtype=torch.long), 2)
    assert m.dtype == torch.float32


def test_routing_output_shapes_contract():
    r = _router(K=3, k=2)
    out = r(torch.randn(2, 5, 16))
    assert isinstance(out, RoutingOutput)
    assert out.gates.shape == out.mask.shape == (2, 5, 3)
    assert out.indices.shape == (2, 5, 2)
