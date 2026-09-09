"""Specialization score (spec §7.2): mutual information between expert
choice and token class, with a permutation-test significance level.
"""

from __future__ import annotations

import numpy as np


def mutual_information(assignments: np.ndarray, classes: np.ndarray) -> float:
    """Empirical MI (nats) between two discrete aligned arrays."""
    a = np.asarray(assignments).ravel()
    c = np.asarray(classes).ravel()
    if a.shape != c.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {c.shape}")
    n = a.shape[0]
    if n == 0:
        return 0.0
    a_vals, a_inv = np.unique(a, return_inverse=True)
    c_vals, c_inv = np.unique(c, return_inverse=True)
    joint = np.zeros((len(a_vals), len(c_vals)))
    np.add.at(joint, (a_inv, c_inv), 1.0 / n)
    pa = joint.sum(axis=1, keepdims=True)
    pc = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = joint * np.log(joint / (pa @ pc))
    return float(np.nansum(terms))


def specialization_score(
    assignments: np.ndarray,
    classes: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 0,
) -> dict:
    """MI plus a permutation null: p = fraction of shuffled MIs ≥ observed."""
    observed = mutual_information(assignments, classes)
    rng = np.random.default_rng(seed)
    flat_classes = np.asarray(classes).ravel()
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        null[i] = mutual_information(assignments, rng.permutation(flat_classes))
    p_value = float((1 + (null >= observed).sum()) / (n_permutations + 1))
    return {
        "mi": observed,
        "p_value": p_value,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
    }


def surprise_decile_specialization(
    assignments: np.ndarray,
    surprise: np.ndarray,
    num_experts: int = 2,
    num_bins: int = 10,
) -> dict:
    """Compute empirical expert selection probabilities per surprise decile.

    Tests the mechanistic hypothesis: does token novelty (surprise) monotonically
    steer routing toward associative memory (e.g. GDR) vs state-space decay (SSD)?

    Returns:
        bin_edges: list of float quantile boundaries
        expert_probabilities: [num_bins, num_experts] matrix of P(expert | bin)
        monotonicity_spearman: per-expert rank correlation across deciles
    """
    a = np.asarray(assignments).ravel()
    s = np.asarray(surprise).ravel()
    if a.shape != s.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {s.shape}")
    if a.size == 0:
        return {
            "bin_edges": [],
            "expert_probabilities": [],
            "monotonicity_spearman": [0.0] * num_experts,
        }

    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(s, quantiles)
    edges[0] = edges[0] - 1e-6
    edges[-1] = edges[-1] + 1e-6

    bin_indices = np.digitize(s, edges[1:-1])
    probs = np.zeros((num_bins, num_experts), dtype=np.float64)

    for b in range(num_bins):
        mask = bin_indices == b
        n_b = mask.sum()
        if n_b > 0:
            for e in range(num_experts):
                probs[b, e] = float((a[mask] == e).sum()) / float(n_b)
        else:
            probs[b, :] = 1.0 / num_experts

    bin_ranks = np.arange(num_bins, dtype=np.float64)
    correlations = []
    for e in range(num_experts):
        p_e = probs[:, e]
        if p_e.std() == 0 or bin_ranks.std() == 0:
            correlations.append(0.0)
        else:
            r = float(np.corrcoef(bin_ranks, p_e)[0, 1])
            correlations.append(0.0 if np.isnan(r) else r)

    return {
        "bin_edges": edges.tolist(),
        "expert_probabilities": probs.tolist(),
        "monotonicity_spearman": correlations,
    }
