"""Analysis suite tests (spec §7 — the primary scientific deliverable)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from mom.analysis import generate_report
from mom.analysis.composition import reference_composition, time_averaged_utilization
from mom.analysis.dynamics import (
    accuracy_trajectory,
    entropy_trajectory,
    pearson,
    specialization_onset,
)
from mom.analysis.heatmaps import routing_assignments
from mom.analysis.knockout import knockout_evaluation, mqar_accuracy_metric
from mom.analysis.specialization import mutual_information, specialization_score
from mom.baselines import build_model
from mom.config import MoMConfig
from mom.tasks.mqar import MQARConfig, make_mqar_batch


def _model(seed=0, **ov):
    cfg = MoMConfig(
        hidden_dim=16,
        num_heads=2,
        num_layers=2,
        ssd_state_dim=8,
        scan_backend="reference",
        **ov,
    )
    torch.manual_seed(seed)
    return build_model("mom", cfg, vocab_size=64)


# ---------------------------------------------------------------------------
# specialization (MI between expert choice and token class)
# ---------------------------------------------------------------------------


def test_mi_independent_is_zero():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 4000)
    c = rng.integers(0, 4, 4000)
    assert mutual_information(a, c) < 0.01


def test_mi_perfect_dependence_equals_entropy():
    c = np.tile([0, 1, 2, 3], 1000)
    mi = mutual_information(c % 2, c)
    # classes uniform over 4 ⇒ H = log 4; assignment = c % 2 keeps log 2 of it…
    # MI must equal the entropy of the coarser variable: log 2.
    assert mi == pytest.approx(np.log(2), abs=1e-6)


def test_mi_antisymmetric_tolerance():
    rng = np.random.default_rng(1)
    c = rng.integers(0, 3, 3000)
    a = (c + rng.integers(0, 2, 3000)) % 3
    assert mutual_information(a, c) == pytest.approx(mutual_information(c, a), abs=1e-9)


def test_specialization_score_significant_vs_null():
    rng = np.random.default_rng(2)
    c = rng.integers(0, 2, 2000)
    a_dep = (c + (rng.random(2000) < 0.05)).astype(int) % 2  # 95% agreement
    a_ind = rng.integers(0, 2, 2000)
    sig = specialization_score(a_dep, c, n_permutations=200, seed=0)
    null = specialization_score(a_ind, c, n_permutations=200, seed=0)
    assert sig["mi"] > 0.5 and sig["p_value"] < 0.05
    assert null["p_value"] > 0.05


# ---------------------------------------------------------------------------
# heatmaps
# ---------------------------------------------------------------------------


def test_routing_assignments_match_router_indices():
    model = _model().eval()
    ids = torch.randint(0, 64, (2, 24))
    with torch.no_grad():
        out = model(ids)
    assign = routing_assignments(model, ids)
    assert assign.shape == (2, 2, 24)  # (layers, B, T)
    for layer in range(2):
        assert np.array_equal(assign[layer], out["routings"][layer].indices[..., 0].numpy())


# ---------------------------------------------------------------------------
# knockout
# ---------------------------------------------------------------------------


def test_knockout_changes_output_and_reports_deltas():
    model = _model().eval()
    task = MQARConfig(vocab_size=64, num_pairs=6, seq_len=32)
    ids, labels = make_mqar_batch(task, 4, torch.Generator().manual_seed(3))
    report = knockout_evaluation(model, ids, labels, mqar_accuracy_metric, experts=("ssd", "gdr"))
    assert "baseline" in report and "ssd" in report and "gdr" in report
    for name in ("ssd", "gdr"):
        assert "accuracy" in report[name] and "delta" in report[name]
        assert report[name]["delta"] == pytest.approx(
            report["baseline"]["accuracy"] - report[name]["accuracy"], abs=1e-9
        )
    # knocking out an expert actually changes the logits
    with torch.no_grad():
        base = model(ids)["logits"]
        ko = model(ids, knockout={0: {"gdr"}, 1: {"gdr"}})["logits"]
    assert not torch.allclose(base, ko)


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------


def test_time_averaged_utilization_matches_counts():
    model = _model().eval()
    ids = torch.randint(0, 64, (2, 32))
    with torch.no_grad():
        out = model(ids)
    util = time_averaged_utilization(out["routings"])
    assert util.shape == (2, 2)
    for layer, r in enumerate(out["routings"]):
        expected = r.mask.mean(dim=(0, 1)).numpy()
        np.testing.assert_allclose(util[layer], expected, rtol=1e-6)


def test_reference_composition_3_to_1():
    ref = reference_composition(ratio=(3, 1))
    np.testing.assert_allclose(ref, [0.75, 0.25])


# ---------------------------------------------------------------------------
# dynamics
# ---------------------------------------------------------------------------


def _history():
    layers = [
        {"entropy": 0.9, "utilization": [0.5, 0.5]},
        {"entropy": 0.7, "utilization": [0.6, 0.4]},
        {"entropy": 0.3, "utilization": [0.8, 0.2]},
        {"entropy": 0.25, "utilization": [0.85, 0.15]},
    ]
    return [
        {"step": i, "accuracy": 0.2 + 0.2 * i, "layers": [layers[i]]}
        for i in range(4)
    ]


def test_trajectories_from_history():
    h = _history()
    np.testing.assert_allclose(entropy_trajectory(h, layer=0), [0.9, 0.7, 0.3, 0.25])
    np.testing.assert_allclose(accuracy_trajectory(h), [0.2, 0.4, 0.6, 0.8])


def test_pearson_hand_computed():
    assert pearson(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])) == pytest.approx(1.0)
    assert pearson(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])) == pytest.approx(-1.0)


def test_specialization_onset_detects_drop():
    h = _history()
    # onset = first eval where entropy < 50% of its initial value
    assert specialization_onset(h, layer=0, drop_fraction=0.5) == 2
    assert specialization_onset(h, layer=0, drop_fraction=0.1) is None


# ---------------------------------------------------------------------------
# versioned report
# ---------------------------------------------------------------------------


def test_generate_report_writes_versioned_artifacts(tmp_path):
    model = _model().eval()
    task = MQARConfig(vocab_size=64, num_pairs=6, seq_len=32)
    ids, labels = make_mqar_batch(task, 2, torch.Generator().manual_seed(5))
    report = generate_report(model, ids, labels, _history(), tmp_path)
    assert report["version"] >= 1
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["version"] == report["version"]
    for key in ("heatmaps", "composition", "knockout", "dynamics"):
        assert key in data
    assert (tmp_path / "routing_heatmaps.npy").exists()


def test_mqar_token_classes_align_with_layout():
    cfg = MQARConfig(vocab_size=64, num_pairs=4, seq_len=24)
    ids, labels, classes = make_mqar_batch(
        cfg, 2, torch.Generator().manual_seed(0), return_classes=True
    )
    # classes: 0=filler, 1=key, 2=value, 3=query
    n = cfg.num_pairs
    assert (classes[:, : 2 * n : 2] == 1).all()
    assert (classes[:, 1 : 2 * n : 2] == 2).all()
    scored = labels != -100
    # the token *preceding* each scored position is a query
    assert (classes[:, :-1][scored[:, 1:]] == 3).all()
