"""Causality / leakage + wiring tests for the layer-local SurprisePredictor.

The predictor is strictly time-causal: ``surprise_t`` is a deterministic function
of ``x_{t-1}, x_t`` (in eval mode, where the running mu/sigma buffers are frozen).
These tests prove no future information leaks into ``surprise_t``, that the
online/EMA separation holds, and that SGMSBlock wires the layer-local predictor in
(design (b)) with the external ``surprise`` argument acting as an override.
Pure unit tests, no GPU required.
"""

import torch
import torch.nn.functional as F
from sgms.block import SGMSBlock
from sgms.config import SGMSConfig
from sgms.model import SGMSLM
from sgms.surprise import SurprisePredictor


def test_no_leakage_from_future():
    pred = SurprisePredictor(hidden_dim=16).eval()
    torch.manual_seed(0)
    x = torch.randn(2, 10, 16)
    s1 = pred(x)
    x2 = x.clone()
    x2[:, 5:, :] = torch.randn_like(x2[:, 5:, :])  # perturb t>=5 only
    s2 = pred(x2)
    assert torch.allclose(s1[:, :5], s2[:, :5], atol=1e-6)  # t<5 unaffected
    assert not torch.allclose(s1[:, 5:], s2[:, 5:])


def test_strict_shift_causality():
    """Changing x_t changes surprise_t (+1 via pred), never the past."""
    pred = SurprisePredictor(hidden_dim=16).eval()
    torch.manual_seed(0)
    x = torch.randn(1, 6, 16)
    s0 = pred(x)
    x_mod = x.clone()
    x_mod[:, 3] += 5.0
    s1 = pred(x_mod)
    assert torch.equal(s0[0, :3], s1[0, :3])  # strictly past: identical
    assert (s0[0, 3] != s1[0, 3]).item()  # uses x_3
    assert (s0[0, 4] != s1[0, 4]).item()  # predicts from x_3


def test_shape_and_range():
    pred = SurprisePredictor(hidden_dim=16, surprise_max=5.0).eval()
    x = torch.randn(3, 7, 16)
    surp = pred(x)
    assert tuple(surp.shape) == (3, 7)
    assert torch.isfinite(surp).all()
    # signed/symmetric clamp: values live in [-max, max]
    assert surp.min() >= -5.0 and surp.max() <= 5.0


def test_signed_surprise_is_centered_when_mu_trained():
    """Once mu tracks abs_diff (many training steps), surprise is centered ~0
    (can be negative) — that is what lets a per-expert weight perturb rather
    than bias the router."""
    pred = SurprisePredictor(hidden_dim=16).train()
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16)
    # Let the running mu/sigma stats converge toward the batch's abs_diff mean.
    # (ema_decay 0.99 => time constant ~100; loop well past it for full convergence)
    for _ in range(3000):
        pred(x)
    centered = pred(x)
    assert abs(centered.mean().item()) < 1e-2  # centered ~0 after convergence


def test_eval_determinism_and_frozen_buffers():
    pred = SurprisePredictor(hidden_dim=16).eval()
    x = torch.randn(2, 5, 16)
    s1 = pred(x)
    s2 = pred(x)
    assert torch.equal(s1, s2)
    assert pred.mu.item() == 0.0 and pred.sigma.item() == 1.0  # untouched in eval


def test_training_updates_stat_buffers():
    pred = SurprisePredictor(hidden_dim=16).train()
    torch.manual_seed(0)
    pred(torch.randn(2, 5, 16))
    assert pred.mu.item() != 0.0  # running mean moved off init


def test_predict_online_trains_online_but_not_ema():
    """Aux MSE on predict_online backprops to online, never to the EMA copy."""
    pred = SurprisePredictor(hidden_dim=8).train()
    torch.manual_seed(0)
    x = torch.randn(2, 6, 8)
    target = x.detach()
    loss = F.mse_loss(pred.predict_online(x), target)
    loss.backward()
    for name, prm in pred.online.named_parameters():
        assert prm.grad is not None and prm.grad.abs().sum() > 0, name
    for prm in pred.ema.parameters():
        assert prm.grad is None  # stable detached baseline


def test_update_ema_blends():
    pred = SurprisePredictor(hidden_dim=16)
    before = [sh.clone() for sh in pred.ema.parameters()]
    with torch.no_grad():
        for prm in pred.online.parameters():
            prm.add_(0.1)
    pred.update_ema()
    assert any(not torch.equal(a, b) for a, b in zip(before, pred.ema.parameters()))


def test_surprise_forward_does_not_backprop_into_ema_baseline():
    """forward() reads surprise solely from the detached EMA baseline."""
    pred = SurprisePredictor(hidden_dim=8)
    x = torch.randn(2, 5, 8, requires_grad=True)
    surp = pred(x)  # uses self.ema(x_prev) only; no online params involved
    surp.sum().backward()
    for prm in pred.online.parameters():
        assert prm.grad is None
    assert x.grad is not None  # surprise is differentiable w.r.t. x


def _block_cfg(use_predictor: bool):
    return SGMSConfig(
        hidden_dim=16,
        num_heads=2,
        num_layers=1,
        ssd_state_dim=8,
        delta_chunk_size=4,
        scan_backend="reference",
        use_surprise_predictor=use_predictor,
        router_surprise_scale=1.0,
    )


def test_block_predictor_present_when_enabled():
    assert SGMSBlock(_block_cfg(True), 0).surprise_predictor is not None
    assert SGMSBlock(_block_cfg(False), 0).surprise_predictor is None


def test_block_wiring_runs_with_internal_predictor_and_override_wins():
    torch.manual_seed(0)
    block = SGMSBlock(_block_cfg(True), 0).eval()
    with torch.no_grad():
        block.router.surprise_weight.copy_(torch.tensor([1.0, -1.0]))
    x = torch.randn(2, 8, 16)
    # internal predictor drives routing (no external surprise given)
    y_in, _, rout_in = block(x)
    assert torch.isfinite(y_in).all() and rout_in.indices.shape == (2, 8, 1)
    # an explicit external surprise overrides the internal predictor. If the
    # external arg were ignored, both calls would use the same internal
    # predictor and produce identical logits; difference proves the override
    # is consumed on the surprise path.
    ext = torch.linspace(0.0, 5.0, 2 * 8).view(2, 8)
    _, _, rout_ext = block(x, surprise=ext)
    assert not torch.equal(rout_in.logits, rout_ext.logits)


def test_predictor_actually_learns():
    """The online head must genuinely reduce prediction error on a learnable
    (periodic) sequence — guard against the 'predictor never trained' failure
    where surprise would be a fixed random projection."""
    torch.manual_seed(0)
    D, T, B = 16, 48, 4
    # periodic, hence predictable from x_{t-1}: P can fit it and MSE drops.
    t = torch.arange(T).float()
    x = torch.stack([torch.sin(0.5 * t - 0.3 * i) for i in range(D)], dim=-1)
    x = x.unsqueeze(0).expand(B, T, D)
    pred = SurprisePredictor(hidden_dim=D, predictor_hidden_dim=32).train()
    opt = torch.optim.AdamW(pred.online.parameters(), lr=1e-2)

    def mse():
        return F.mse_loss(pred.predict_online(x), x.detach())

    before = mse().item()
    for _ in range(50):
        opt.zero_grad()
        loss = mse()
        loss.backward()
        opt.step()
        pred.update_ema()
    after = mse().item()
    assert after < before * 0.5, f"predictor barely learns: {before:.4f} -> {after:.4f}"


def test_model_returns_pred_loss_when_enabled():
    """SGMSLM forward accumulates an aux pred_loss when the predictor is on."""
    torch.manual_seed(0)
    cfg = _block_cfg(True)  # use_surprise_predictor=True
    model = SGMSLM(cfg, vocab_size=16)  # training mode by default
    ids = torch.randint(0, 16, (2, 8))
    out = model(ids)
    assert "pred_loss" in out
    assert torch.isfinite(out["pred_loss"]) and out["pred_loss"].item() >= 0


def test_model_does_not_compute_pred_loss_in_eval():
    torch.manual_seed(0)
    model = SGMSLM(_block_cfg(True), vocab_size=16).eval()
    ids = torch.randint(0, 16, (2, 8))
    out = model(ids)
    assert out["pred_loss"].item() == 0.0  # gated on self.training


def test_config_loads_and_freezes_surprise_weight():
    """Stage-0 probe wiring: config sets the per-expert weight and freezes it."""
    cfg = SGMSConfig(
        hidden_dim=16,
        num_heads=2,
        num_layers=1,
        ssd_state_dim=8,
        delta_chunk_size=4,
        scan_backend="reference",
        router_surprise_scale=1.0,
        router_surprise_weight=(1.0, -1.0),
        freeze_surprise_weight=True,
    )
    block = SGMSBlock(cfg, 0)
    w = block.router.surprise_weight
    assert torch.allclose(w.detach(), torch.tensor([1.0, -1.0]))
    assert not w.requires_grad  # frozen for the probe


def test_config_surprise_weight_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        SGMSConfig(
            hidden_dim=16,
            num_heads=2,
            num_layers=1,
            router_surprise_weight=(1.0,),  # 1 != num_experts=2
        )
