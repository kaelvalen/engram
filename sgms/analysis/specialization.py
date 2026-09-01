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
