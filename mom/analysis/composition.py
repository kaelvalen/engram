"""Learned composition (spec §7.4): time-averaged expert utilization per
layer vs. the hand-tuned 3:1 reference line.
"""

from __future__ import annotations

import numpy as np


def time_averaged_utilization(routings) -> np.ndarray:
    """Per-layer fraction of tokens routed to each expert → (L, K)."""
    return np.stack([r.mask.float().mean(dim=(0, 1)).cpu().numpy() for r in routings], axis=0)


def reference_composition(ratio: tuple[float, ...] = (3, 1)) -> np.ndarray:
    """Normalised fixed-ratio reference line (default PRISM 3:1)."""
    r = np.asarray(ratio, dtype=np.float64)
    return r / r.sum()


def composition_summary(routings, ratio: tuple[float, ...] = (3, 1)) -> dict:
    """Utilization per layer, the reference line, and mean abs deviation."""
    util = time_averaged_utilization(routings)
    ref = reference_composition(ratio)
    return {
        "utilization_per_layer": util.tolist(),
        "reference": ref.tolist(),
        "mean_abs_deviation_from_reference": float(np.abs(util - ref).mean()),
    }
