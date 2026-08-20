"""Causality / leakage tests for the layer-local HiddenSurprisePredictor (§7.3).

The predictor is strictly time-causal: ``surprise_t`` is a deterministic function
of ``x_{t-1}, x_t`` (in eval mode, where the running mu/sigma buffers are frozen).
These tests prove no future information leaks into ``surprise_t`` and that the
online/EMA separation works. Pure unit tests, no GPU required.
"""

import torch
from mom.surprise import HiddenSurprisePredictor, SurprisePredictorConfig


def _cfg(hidden_dim=16):
    return SurprisePredictorConfig(hidden_dim=hidden_dim)


def test_no_future_leakage():
    """Perturbing strictly-future positions must not change surprise[:, :t0]."""
    p = HiddenSurprisePredictor(_cfg())
    p.eval()
    torch.manual_seed(0)
    x = torch.randn(2, 10, 16)
    surp_full, _ = p(x)

    x_bad = x.clone()
    x_bad[:, 5:] += 1.0  # perturb strictly-future tokens
    surp_bad, _ = p(x_bad)

    # future selection (indices 0..4) is untouched
    assert torch.equal(surp_full[:, :5], surp_bad[:, :5])
    # positions that see the perturbation do change
    assert not torch.equal(surp_full[:, 5:], surp_bad[:, 5:])


def test_strict_shift_causality():
    """Changing x_t changes surprise_t (and surprise_{t+1} via pred), but not
    surprise_{t-1}; pred at t uses only x_{t-1}."""
    p = HiddenSurprisePredictor(_cfg())
    p.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 6, 16)
    s0, _ = p(x)

    x_mod = x.clone()
    x_mod[:, 3] += 5.0  # perturb position 3
    s1, _ = p(x_mod)

    # positions strictly before the changed token are identical
    assert torch.equal(s0[0, :3], s1[0, :3])
    # surprise at t=3 uses x_3 -> changes
    assert (s0[0, 3] != s1[0, 3]).item()
    # surprise at t=4 predicts from x_3 -> also changes
    assert (s0[0, 4] != s1[0, 4]).item()


def test_shape_and_range():
    cfg = _cfg()
    p = HiddenSurprisePredictor(cfg)
    p.eval()
    x = torch.randn(3, 7, 16)
    surp, aux = p(x)
    assert tuple(surp.shape) == (3, 7)
    assert torch.isfinite(surp).all()
    assert surp.min() >= 0
    assert surp.max() <= cfg.surprise_max
    assert tuple(aux["x_hat"].shape) == tuple(x.shape)


def test_eval_determinism():
    """Eval mode must not update the running stat buffers (deterministic)."""
    p = HiddenSurprisePredictor(_cfg())
    p.eval()
    x = torch.randn(2, 5, 16)
    s1, _ = p(x)
    s2, _ = p(x)
    assert torch.equal(s1, s2)
    # buffers must be exactly their init values (untouched in eval)
    assert p.mu.item() == 0.0 and p.sigma.item() == 1.0


def test_training_updates_stat_buffers():
    p = HiddenSurprisePredictor(_cfg())
    p.train()
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    p(x)
    assert p.mu.item() != 0.0  # running mean moved off the init value


def test_pred_loss_trains_online_but_not_ema():
    """pred_loss backprops to the online predictor, never to the EMA shadows."""
    p = HiddenSurprisePredictor(_cfg(hidden_dim=8))
    p.train()
    x = torch.randn(2, 6, 8)
    _, aux = p(x)
    aux["pred_loss"].backward()
    for name, prm in p.predictor.named_parameters():
        assert prm.grad is not None and prm.grad.abs().sum() > 0, name
    for sh in p._ema_shadows:
        assert sh.grad is None  # detached stable baseline, no grad


def test_update_ema_blends_shadows():
    p = HiddenSurprisePredictor(_cfg())
    before = [sh.clone() for sh in p._ema_shadows]
    # move online weights away so the blend has an effect
    with torch.no_grad():
        for prm in p.predictor.parameters():
            prm.add_(0.1)
    p.update_ema()
    moved = any(not torch.equal(a, b) for a, b in zip(before, p._ema_shadows))
    assert moved
