"""Routing heatmaps (spec §7.1): per layer, token position × expert assignment."""

from __future__ import annotations

import numpy as np
import torch


def assignments_from_routings(routings: list) -> np.ndarray:
    """Selected expert per (layer, batch, position) from precomputed routings."""
    return np.stack([r.gates.argmax(-1).cpu().numpy() for r in routings], axis=0)


@torch.no_grad()
def routing_assignments(model, input_ids: torch.Tensor) -> np.ndarray:
    """Selected expert per (layer, batch, position): int array (L, B, T).

    For k>1 the highest-gated expert is taken as the assignment.
    """
    out = model(input_ids)
    return assignments_from_routings(out["routings"])


def heatmap_utilization(assignments: np.ndarray, num_experts: int) -> np.ndarray:
    """Per-layer expert histogram of a (L, B, T) assignment array → (L, K)."""
    L = assignments.shape[0]
    util = np.zeros((L, num_experts), dtype=np.float64)
    for layer in range(L):
        counts = np.bincount(assignments[layer].ravel(), minlength=num_experts)
        util[layer] = counts / counts.sum()
    return util
