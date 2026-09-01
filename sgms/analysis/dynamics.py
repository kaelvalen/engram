"""Routing dynamics (spec §7.5): entropy/utilization trajectories over
training, specialization onset, and its correlation with task metrics.
"""

from __future__ import annotations

import numpy as np


def entropy_trajectory(history: list[dict], layer: int = 0) -> np.ndarray:
    """Routing entropy per eval record for one layer."""
    return np.array([rec["layers"][layer]["entropy"] for rec in history], dtype=np.float64)


def utilization_trajectory(history: list[dict], layer: int = 0) -> np.ndarray:
    """Per-eval expert utilization for one layer → (num_evals, K)."""
    return np.array([rec["layers"][layer]["utilization"] for rec in history], dtype=np.float64)


def accuracy_trajectory(history: list[dict]) -> np.ndarray:
    """Task accuracy per eval record."""
    return np.array([rec["accuracy"] for rec in history], dtype=np.float64)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation; 0.0 when either series is constant."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("series must be non-empty and equally long")
    xs, ys = x.std(), y.std()
    if xs == 0 or ys == 0:
        return 0.0
    return float(((x - x.mean()) * (y - y.mean())).mean() / (xs * ys))


def specialization_onset(
    history: list[dict], layer: int = 0, drop_fraction: float = 0.5
) -> int | None:
    """First eval index where routing entropy fell below
    ``drop_fraction`` of its initial value — the specialization onset."""
    entropy = entropy_trajectory(history, layer)
    if entropy.size == 0:
        return None
    threshold = entropy[0] * drop_fraction
    below = np.nonzero(entropy < threshold)[0]
    return int(below[0]) if below.size else None


def dynamics_summary(history: list[dict], layer: int = 0) -> dict:
    """Trajectories + onset + onset/accuracy correlation (§7.5)."""
    entropy = entropy_trajectory(history, layer)
    accuracy = accuracy_trajectory(history)
    return {
        "entropy": entropy.tolist(),
        "accuracy": accuracy.tolist(),
        "utilization": utilization_trajectory(history, layer).tolist(),
        "specialization_onset_eval": specialization_onset(history, layer),
        "entropy_accuracy_correlation": pearson(entropy, accuracy),
    }
