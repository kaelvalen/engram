"""Training wiring: MQAR task, B1–B5 baselines, and the spike training loop."""

from __future__ import annotations

import torch
from mom.baselines import HybridLM, build_model, layer_pattern
from mom.config import MoMConfig
from mom.model import MoMLM
from mom.tasks.mqar import MQARConfig, make_mqar_batch
from mom.train import evaluate, train_one


def _task_cfg(**kw):
    base = dict(vocab_size=64, num_pairs=6, seq_len=32)
    base.update(kw)
    return MQARConfig(**base)


# ---------------------------------------------------------------------------
# MQAR task
# ---------------------------------------------------------------------------


def test_mqar_batch_shapes_and_masking():
    cfg = _task_cfg()
    g = torch.Generator().manual_seed(0)
    ids, labels = make_mqar_batch(cfg, 4, g)
    assert ids.shape == labels.shape == (4, 32)
    assert ids.dtype == torch.long
    # scored positions = one per query
    assert (labels != -100).sum() == 4 * cfg.num_pairs


def test_mqar_labels_recall_correct_values():
    cfg = _task_cfg(num_pairs=4, seq_len=24)
    g = torch.Generator().manual_seed(1)
    ids, labels = make_mqar_batch(cfg, 8, g)
    for b in range(8):
        seq, lab = ids[b], labels[b]
        # learn the presented pairs from the pair region
        pairs = {}
        for t in range(2 * cfg.num_pairs - 1, -1, -2):
            pairs[int(seq[t - 1])] = int(seq[t])
        scored = (lab != -100).nonzero().flatten()
        for p in scored:
            query_token = int(seq[p - 1])  # token at p-1 is the query
            assert query_token in pairs
            assert int(lab[p]) == pairs[query_token]


def test_mqar_determinism_same_seed():
    cfg = _task_cfg()
    a = make_mqar_batch(cfg, 2, torch.Generator().manual_seed(7))
    b = make_mqar_batch(cfg, 2, torch.Generator().manual_seed(7))
    c = make_mqar_batch(cfg, 2, torch.Generator().manual_seed(8))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[0], c[0])


def test_mqar_tokens_within_vocab():
    cfg = _task_cfg()
    ids, labels = make_mqar_batch(cfg, 4, torch.Generator().manual_seed(0))
    assert ids.min() >= 0 and ids.max() < cfg.vocab_size
    scored = labels[labels != -100]
    assert scored.min() >= 0 and scored.max() < cfg.vocab_size


def test_mqar_rejects_too_short_context():
    import pytest

    with pytest.raises(ValueError):
        make_mqar_batch(_task_cfg(seq_len=10), 1, torch.Generator().manual_seed(0))


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def test_b1_pattern_is_3_to_1():
    pat = layer_pattern("B1", 8)
    assert pat == ["ssd", "ssd", "ssd", "gdr", "ssd", "ssd", "ssd", "gdr"]


def test_b2_b3_patterns_single_primitive():
    assert layer_pattern("B2", 4) == ["ssd"] * 4
    assert layer_pattern("B3", 4) == ["gdr"] * 4


def test_unknown_baseline_rejected():
    import pytest

    with pytest.raises(ValueError):
        build_model("B9", MoMConfig(), vocab_size=32)


def _cfg():
    return MoMConfig(
        hidden_dim=16, num_heads=2, num_layers=2, ssd_state_dim=8, scan_backend="reference"
    )


def test_baselines_forward_shapes():
    for kind in ["B1", "B2", "B3"]:
        m = build_model(kind, _cfg(), vocab_size=32)
        out = m(torch.randint(0, 32, (2, 16)))
        assert out["logits"].shape == (2, 16, 32)


def test_b4_b5_are_mom_with_frozen_routers():
    b4 = build_model("B4", _cfg(), vocab_size=32)
    b5 = build_model("B5", _cfg(), vocab_size=32)
    assert isinstance(b4, MoMLM) and isinstance(b5, MoMLM)
    assert b4.cfg.router_mode == "uniform" and b5.cfg.router_mode == "random"
    for blk in list(b4.blocks) + list(b5.blocks):
        assert not blk.router.weight.requires_grad


def test_hybrid_lm_state_passing_runs():
    m = HybridLM("B1", _cfg(), vocab_size=32).eval()
    ids = torch.randint(0, 32, (1, 20))
    full = m(ids)["logits"]
    p1 = m(ids[:, :10])
    p2 = m(ids[:, 10:], p1["states"])
    torch.testing.assert_close(
        torch.cat([p1["logits"], p2["logits"]], 1), full, rtol=2e-3, atol=2e-3
    )


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------


def _train_config(kind="mom", steps=8, seed=0):
    return {
        "experiment": "test",
        "model": {
            "kind": kind,
            "hidden_dim": 32,
            "num_heads": 2,
            "num_layers": 2,
            "ssd_state_dim": 8,
            "scan_backend": "reference",
        },
        "task": {"name": "mqar", "vocab_size": 64, "num_pairs": 6, "seq_len": 32},
        "optim": {
            "steps": steps,
            "batch_size": 4,
            "lr": 3e-3,
            "weight_decay": 0.0,
            "warmup_frac": 0.1,
            "grad_clip": 1.0,
            "bf16": False,
        },
        "eval_every": 4,
    }


def test_train_one_returns_finite_metrics_and_stats():
    out = train_one(_train_config(), seed=0)
    hist = out["history"]
    assert len(hist) == 8
    final = hist[-1]
    assert torch.isfinite(torch.tensor(final["task_loss"]))
    assert "accuracy" in final and 0.0 <= final["accuracy"] <= 1.0
    # routing stats are first-class metrics (spec §6.1)
    assert "min_utilization" in final
    assert len(final["layers"]) == 2
    assert "entropy" in final["layers"][0]


def test_train_determinism_same_seed():
    a = train_one(_train_config(steps=4), seed=0)
    b = train_one(_train_config(steps=4), seed=0)
    la = [h["task_loss"] for h in a["history"]]
    lb = [h["task_loss"] for h in b["history"]]
    assert la == lb


def test_train_baselines_run_without_router_stats():
    out = train_one(_train_config(kind="B1", steps=4), seed=0)
    assert len(out["history"]) == 4
    assert out["history"][-1]["layers"] == []


def test_evaluate_accuracy_range():
    model = build_model("mom", _cfg(), vocab_size=64)
    task = _task_cfg()
    acc = evaluate(model, task, batch_size=4, seed=99)
    assert 0.0 <= acc <= 1.0


def test_evaluate_scores_cheating_model_perfectly():
    """Off-by-one regression: logits[t] predicts token t+1, so a model that
    places all mass on the correct value AT the query position must score 1.0."""
    task = _task_cfg()
    ids, labels = make_mqar_batch(task, 4, torch.Generator().manual_seed(999))

    class CheatModel(torch.nn.Module):
        def forward(self, ids):
            logits = torch.full((ids.shape[0], ids.shape[1], task.vocab_size), -10.0)
            tgt = labels[:, 1:]
            pos = (tgt != -100).nonzero()
            logits[pos[:, 0], pos[:, 1] - 1, tgt[tgt != -100]] = 10.0
            return {"logits": logits}

    assert evaluate(CheatModel(), task, batch_size=4, seed=999) == 1.0
