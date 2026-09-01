"""Stability objectives (spec §3.7): Switch-style load balancing + router z-loss.

L_bal = K · Σ_e f_e · P_e     f_e: hard routed fraction (non-differentiable)
                              P_e: mean softmax probability (differentiable)
L_z   = mean_t ( logsumexp(z_t) )²
L     = L_task + λ_bal · L_bal + λ_z · L_z
"""

from __future__ import annotations

import math

import torch
from sgms.losses import load_balancing_loss, router_z_loss, sgms_auxiliary_loss
from sgms.router import RoutingOutput


def _routing(mask, probs=None, logits=None, gates=None):
    B, T, K = mask.shape
    idx = mask.topk(1, dim=-1).indices
    return RoutingOutput(
        gates=mask.clone() if gates is None else gates,
        mask=mask.float(),
        indices=idx,
        logits=logits,
        probs=probs,
    )


def _uniform_routing(B=2, T=4, K=2, requires_grad=False):
    probs = torch.full((B, T, K), 1 / K, requires_grad=requires_grad)
    mask = torch.zeros(B, T, K)
    mask[..., 0] = 1  # f = (1, 0) while P = (1/K, 1/K)
    return _routing(mask, probs=probs)


# ---------------------------------------------------------------------------
# load balancing
# ---------------------------------------------------------------------------


def test_lbal_uniform_probs_hard_counts():
    # f = (1, 0), P = (1/2, 1/2) ⇒ K·Σ f·P = 2·(1·½ + 0·½) = 1
    out = _uniform_routing()
    loss = load_balancing_loss([out])
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_lbal_fully_collapsed_probs():
    # f = (1, 0), P = (1, 0) ⇒ 2·(1·1 + 0·0) = 2 = K
    B, T, K = 1, 3, 2
    probs = torch.zeros(B, T, K)
    probs[..., 0] = 1.0
    mask = torch.zeros(B, T, K)
    mask[..., 0] = 1
    loss = load_balancing_loss([_routing(mask, probs=probs)])
    torch.testing.assert_close(loss, torch.tensor(float(K)))


def test_lbal_balanced_ideal():
    # f = P = (1/K, …) ⇒ K · Σ (1/K²) = 1
    B, T, K = 1, 8, 4
    mask = torch.zeros(B, T, K)
    mask[0, torch.arange(T), torch.arange(T) % K] = 1  # cyclic hard counts
    probs = torch.full((B, T, K), 1 / K)
    loss = load_balancing_loss([_routing(mask, probs=probs)])
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_lbal_gradient_flows_through_probs_not_counts():
    out = _uniform_routing(requires_grad=True)
    loss = load_balancing_loss([out])
    g = torch.autograd.grad(loss, out.probs)[0]
    # P_e is a mean over B·T tokens ⇒ dL/dP_e-entry = K·f_e/(B·T): expert 0
    # gets 2·1/8 = 0.25, expert 1 gets 0 (f is a non-differentiable count).
    torch.testing.assert_close(g[..., 0], torch.full_like(g[..., 0], 0.25))
    torch.testing.assert_close(g[..., 1], torch.zeros_like(g[..., 1]))


def test_lbal_averages_over_layers():
    a = _uniform_routing()  # loss 1
    b = _uniform_routing()  # loss 1
    loss = load_balancing_loss([a, b])
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_lbal_skips_layers_without_probs():
    a = _uniform_routing()
    no_probs = _routing(torch.ones(1, 2, 2), probs=None)
    loss = load_balancing_loss([a, no_probs])
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_lbal_no_probs_anywhere_returns_zero():
    out = _routing(torch.ones(1, 2, 2), probs=None)
    loss = load_balancing_loss([out])
    torch.testing.assert_close(loss, torch.tensor(0.0))


# ---------------------------------------------------------------------------
# z-loss
# ---------------------------------------------------------------------------


def test_zloss_hand_computed():
    logits = torch.tensor([[[0.0, math.log(2.0)], [1.0, -1.0]]])
    expected = (math.log(1 + 2) ** 2 + math.log(math.e + math.e**-1) ** 2) / 2
    loss = router_z_loss([_routing(torch.ones(1, 2, 2), logits=logits)])
    torch.testing.assert_close(loss, torch.tensor(expected))


def test_zloss_zero_logits_is_logK_squared():
    K = 3
    logits = torch.zeros(2, 5, K)
    loss = router_z_loss([_routing(torch.ones(2, 5, K), logits=logits)])
    torch.testing.assert_close(loss, torch.tensor(math.log(K) ** 2))


def test_zloss_gradient_flows():
    logits = torch.randn(1, 4, 2, requires_grad=True)
    loss = router_z_loss([_routing(torch.ones(1, 4, 2), logits=logits)])
    g = torch.autograd.grad(loss, logits)[0]
    assert g.abs().sum() > 0


def test_zloss_skips_layers_without_logits():
    with_logits = _routing(torch.ones(1, 2, 2), logits=torch.zeros(1, 2, 2))
    without = _routing(torch.ones(1, 2, 2), logits=None)
    loss = router_z_loss([with_logits, without])
    torch.testing.assert_close(loss, torch.tensor(math.log(2) ** 2))


# ---------------------------------------------------------------------------
# combined auxiliary loss
# ---------------------------------------------------------------------------


def test_auxiliary_loss_lambda_scaling():
    B, T, K = 1, 4, 2
    probs = torch.full((B, T, K), 0.5)
    mask = torch.zeros(B, T, K)
    mask[..., 0] = 1
    logits = torch.zeros(B, T, K)
    out = _routing(mask, probs=probs, logits=logits)
    total, parts = sgms_auxiliary_loss([out], lambda_bal=1e-2, lambda_z=1e-3)
    # L_bal = 1, L_z = log(2)²
    expected = 1e-2 * 1.0 + 1e-3 * math.log(2) ** 2
    torch.testing.assert_close(total, torch.tensor(expected))
    torch.testing.assert_close(parts["bal"], torch.tensor(1.0))
    torch.testing.assert_close(parts["z"], torch.tensor(math.log(2) ** 2))


def test_auxiliary_loss_zero_lambdas():
    out = _uniform_routing()
    total, _ = sgms_auxiliary_loss([out], lambda_bal=0.0, lambda_z=0.0)
    torch.testing.assert_close(total, torch.tensor(0.0))


def test_auxiliary_loss_gradient_reaches_router_weight():
    from sgms.router import TokenRouter

    torch.manual_seed(0)
    r = TokenRouter(hidden_dim=8, num_experts=2, top_k=1)
    out = r(torch.randn(2, 6, 8))
    total, _ = sgms_auxiliary_loss([out], lambda_bal=1e-2, lambda_z=1e-3)
    g = torch.autograd.grad(total, r.weight)[0]
    assert g.abs().sum() > 0
