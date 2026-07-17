"""SABER tests: the staged module must run, learn, and stay coherent.

Covers the regression set for the repaired bugs: GRU hidden shape,
InfoNCE geometry, predictor learning path, vectorized memory readout,
disjoint optimizer groups, eval-time buffer hygiene, beta annealing,
backbone head logits, 3-phase trainer, and recovery mechanics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from prism.saber import SABER, SABERBackbone, SABERConfig, SABERTrainer
from prism.saber.diagnostics import SABERRecovery


def _cfg(**kw):
    base = dict(
        encoder_hidden_dim=32,
        encoder_num_layers=2,
        policy_state_dim=16,
        policy_num_layers=2,
        num_memory_slots=8,
        memory_slot_dim=32,
        budget_floor=2,
        budget_max=8,
        budget_alpha=2.0,
        infonce_beta_anneal_steps=4,
        phase1_steps=2,
        phase2_steps=2,
        phase3_steps=100,
    )
    base.update(kw)
    return SABERConfig(**base)


def _saber(cfg=None, seed=0, input_dim=12):
    torch.manual_seed(seed)
    return SABER(cfg or _cfg(), input_dim=input_dim)


class _ToyBackbone(nn.Module):
    """Minimal (B,T,D) → (B,T,D) backbone with the PRISM return contract."""

    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x, states=None):
        return self.lin(x), states


# ---------------------------------------------------------------------------
# forward / shapes
# ---------------------------------------------------------------------------


def test_forward_smoke_two_layer_gru():
    """Regression: GRU with num_layers=2 crashed on hidden shape (1,B,H)."""
    s = _saber()
    state, aux = s(torch.randn(2, 7, 12))
    assert state.policy_hidden.shape == (2, 2, 16)  # (num_layers, B, H)


def test_aux_contract_and_shapes():
    s = _saber()
    _, aux = s(torch.randn(2, 7, 12))
    for key in (
        "z_t",
        "s_t",
        "z_hat_t",
        "surprise",
        "budget",
        "infonce_loss",
        "predictor_loss",
        "memory_readout",
        "memory_weights",
    ):
        assert key in aux, key
    assert aux["memory_readout"].shape == (2, 7, 32)
    w = aux["memory_weights"]
    assert w.shape == (2, 7, 8)
    assert (w >= 0).all()
    torch.testing.assert_close(w.sum(-1), torch.ones(2, 7), rtol=1e-5, atol=1e-6)


def test_budget_within_bounds_and_surprise_nonnegative():
    cfg = _cfg()
    s = _saber(cfg)
    _, aux = s(torch.randn(2, 9, 12))
    assert (aux["surprise"] >= 0).all()
    lo = cfg.budget_floor
    hi = cfg.budget_floor + cfg.budget_alpha * cfg.budget_max
    assert (aux["budget"] >= lo).all() and (aux["budget"] <= hi).all()


def test_state_passing_chunked_policy_equals_full():
    """GRU policy hidden carried across chunks ⇒ identical s_t as one pass."""
    s = _saber().eval()
    x = torch.randn(1, 10, 12)
    _, aux_full = s(x)
    st1, aux1 = s(x[:, :4])
    st2, aux2 = s(x[:, 4:], st1)
    torch.testing.assert_close(
        torch.cat([aux1["s_t"], aux2["s_t"]], dim=1),
        aux_full["s_t"],
        rtol=1e-10,
        atol=1e-12,
    )


def test_determinism_same_seed():
    x = torch.randn(1, 6, 12)
    a = _saber(seed=3).eval()(x)[1]["surprise"]
    b = _saber(seed=3).eval()(x)[1]["surprise"]
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# learning paths
# ---------------------------------------------------------------------------


def test_infonce_gradients_reach_policy_and_encoder():
    s = _saber()
    _, aux = s(torch.randn(4, 5, 12))
    loss = aux["infonce_loss"]
    assert torch.isfinite(loss)
    grads = torch.autograd.grad(
        loss,
        [p for p in s.policy.projection_head.parameters()] + [p for p in s.encoder.parameters()],
        allow_unused=True,
    )
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_predictor_loss_trains_online_weight_only():
    s = _saber()
    _, aux = s(torch.randn(2, 5, 12))
    loss = aux["predictor_loss"]
    assert loss >= 0
    g_w = torch.autograd.grad(loss, s.predictor.linear.weight, retain_graph=True)[0]
    assert g_w.abs().sum() > 0
    g_enc = torch.autograd.grad(loss, list(s.encoder.parameters()), allow_unused=True)
    assert all(g is None or g.abs().sum() == 0 for g in g_enc)


def test_ema_updates_only_on_explicit_call():
    s = _saber()
    s.train()
    before = s.predictor.ema_weight.clone()
    s(torch.randn(2, 5, 12))
    assert torch.equal(before, s.predictor.ema_weight)
    with torch.no_grad():
        s.predictor.linear.weight.add_(1.0)
    s.update_ema()
    assert not torch.equal(before, s.predictor.ema_weight)


def test_surprise_buffers_frozen_in_eval():
    s = _saber()
    s.eval()
    mu, sigma = s.surprise.mu.clone(), s.surprise.sigma.clone()
    s(torch.randn(2, 5, 12))
    assert torch.equal(mu, s.surprise.mu) and torch.equal(sigma, s.surprise.sigma)
    s.train()
    s(torch.randn(2, 5, 12))
    assert not torch.equal(mu, s.surprise.mu)


def test_beta_anneals_over_train_forwards():
    cfg = _cfg()
    s = _saber(cfg)
    s.train()
    b0 = float(s.beta)
    for _ in range(cfg.infonce_beta_anneal_steps + 2):
        s(torch.randn(1, 4, 12))
    assert float(s.beta) > b0
    assert float(s.beta) == cfg.infonce_beta_end


def test_param_groups_disjoint_and_complete():
    s = _saber()
    groups = s.get_param_groups()
    seen = set()
    for g in groups:
        for p in g["params"]:
            assert id(p) not in seen
            seen.add(id(p))
    total = sum(1 for _ in s.parameters())
    assert len(seen) == total


# ---------------------------------------------------------------------------
# backbone wrapper / trainer
# ---------------------------------------------------------------------------


def test_backbone_head_returns_logits():
    s = _saber()
    bb = SABERBackbone(s, _ToyBackbone(12), num_classes=5, backbone_out_dim=12)
    y, state, _, aux = bb(torch.randn(2, 6, 12))
    assert y.shape == (2, 5)


def test_backbone_without_head_returns_hidden():
    s = _saber()
    bb = SABERBackbone(s, _ToyBackbone(12))
    y, _, _, _ = bb(torch.randn(2, 6, 12))
    assert y.shape == (2, 6, 12)


def test_trainer_runs_all_three_phases():
    cfg = _cfg()
    s = _saber(cfg)
    bb = SABERBackbone(s, _ToyBackbone(12), num_classes=5, backbone_out_dim=12)
    trainer = SABERTrainer(bb, cfg, torch.device("cpu"))
    batch = {"x": torch.randn(2, 6, 12), "labels": torch.randint(0, 5, (2,))}
    out = [trainer.train_step(batch) for _ in range(5)]
    phases = [o["phase"] for o in out]
    assert phases == [1, 1, 2, 2, 3]
    assert all(torch.isfinite(torch.tensor(o["loss"])) for o in out)
    assert "predictor_loss" in out[-1]


def test_trainer_phase1_freezes_saber_phase2_freezes_backbone():
    cfg = _cfg()
    s = _saber(cfg)
    bb = SABERBackbone(s, _ToyBackbone(12), num_classes=5, backbone_out_dim=12)
    trainer = SABERTrainer(bb, cfg, torch.device("cpu"))
    batch = {"x": torch.randn(2, 6, 12), "labels": torch.randint(0, 5, (2,))}
    trainer.train_step(batch)  # phase 1
    assert not any(p.requires_grad for p in s.policy.parameters())
    assert any(p.requires_grad for p in bb.backbone.parameters())
    trainer.train_step(batch)
    trainer.train_step(batch)  # phase 2
    assert not any(p.requires_grad for p in bb.backbone.parameters())
    assert any(p.requires_grad for p in s.policy.parameters())


# ---------------------------------------------------------------------------
# diagnostics & recovery
# ---------------------------------------------------------------------------


def _recovery_setup():
    cfg = _cfg()
    s = _saber(cfg)
    return cfg, s, SABERRecovery(s, cfg)


def test_recovery_r1_fires_and_resets_predictor():
    cfg, s, rec = _recovery_setup()
    rec.diagnostics.budget_history = [0.1] * (cfg.r1_patience + 1)
    before = s.predictor.ema_weight.clone()
    triggered = rec.check_and_recover({}, step=1)
    assert "R1" in triggered
    assert not torch.equal(before, s.predictor.ema_weight)


def test_recovery_r2_freezes_slot_embeddings():
    cfg, s, rec = _recovery_setup()
    rec.diagnostics.memory_grad_norm_history = [10.0] * (cfg.r2_patience + 1)
    rec.diagnostics.backbone_grad_norm_history = [1.0] * (cfg.r2_patience + 1)
    assert "R2" in rec.check_and_recover({}, step=1)
    assert not s.memory.slot_embeddings.requires_grad


def test_recovery_r3_r4_r5_fire():
    cfg, s, rec = _recovery_setup()
    rec.diagnostics.budget_var_history = [100.0] * (cfg.r3_patience + 1)
    rec.diagnostics.budget_history = [cfg.budget_floor + 0.01] * (cfg.r4_patience + 1)
    rec.diagnostics.infonce_loss_history = [0.001] * 101
    triggered = rec.check_and_recover({}, step=1)
    assert {"R3", "R4", "R5"} <= set(triggered)


def test_recovery_keeps_optimizer_param_references():
    cfg, s, rec = _recovery_setup()
    opt = torch.optim.AdamW(s.get_param_groups())
    ids_before = {id(p) for g in opt.param_groups for p in g["params"]}
    rec.diagnostics.infonce_loss_history = [0.001] * 101  # trips R5
    rec.check_and_recover({}, step=1)
    ids_after = {id(p) for g in opt.param_groups for p in g["params"]}
    assert ids_before == ids_after
    # policy params were reset in place, not replaced
    assert all(id(p) in ids_before for p in s.policy.parameters())
