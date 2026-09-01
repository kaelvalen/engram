"""SGMS analysis suite (spec §7) — the primary scientific deliverable.

Produces a versioned report (report.json + .npy artifacts) covering:
routing heatmaps, specialization MI, expert knockout, learned composition
vs. the 3:1 reference, and routing dynamics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .composition import composition_summary, reference_composition, time_averaged_utilization
from .dynamics import dynamics_summary
from .heatmaps import heatmap_utilization, routing_assignments
from .knockout import knockout_evaluation, mqar_accuracy_metric
from .specialization import mutual_information, specialization_score

REPORT_VERSION = 1

__all__ = [
    "REPORT_VERSION",
    "composition_summary",
    "dynamics_summary",
    "generate_report",
    "heatmap_utilization",
    "knockout_evaluation",
    "mqar_accuracy_metric",
    "mutual_information",
    "reference_composition",
    "routing_assignments",
    "specialization_score",
    "time_averaged_utilization",
]


@torch.no_grad()
def generate_report(
    model,
    ids: torch.Tensor,
    labels: torch.Tensor,
    history: list[dict],
    out_dir,
    token_classes: torch.Tensor | None = None,
    experts: tuple[str, ...] = ("ssd", "gdr"),
    n_permutations: int = 500,
) -> dict:
    """Run the full §7 suite and write a versioned report to ``out_dir``."""
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out = model(ids)
    routings = out["routings"]

    # §7.1 heatmaps
    assignments = routing_assignments(model, ids)  # (L, B, T)
    np.save(out_dir / "routing_heatmaps.npy", assignments)

    # §7.2 specialization (needs token classes, e.g. MQAR key/value/query/filler)
    specialization = None
    if token_classes is not None:
        classes = token_classes.cpu().numpy()
        per_layer = []
        for layer in range(assignments.shape[0]):
            score = specialization_score(
                assignments[layer].ravel(), classes.ravel(), n_permutations=n_permutations
            )
            per_layer.append(score)
        specialization = {"per_layer": per_layer}

    # §7.3 knockout
    knockout = knockout_evaluation(model, ids, labels, mqar_accuracy_metric, experts=experts)

    # §7.4 learned composition vs. 3:1
    composition = composition_summary(routings)

    # §7.5 dynamics (per layer; routing stats exist only on eval records)
    eval_records = [rec for rec in history if "layers" in rec]
    dynamics = (
        {
            f"layer{layer}": dynamics_summary(eval_records, layer)
            for layer in range(len(eval_records[0]["layers"]))
        }
        if eval_records
        else {}
    )

    report = {
        "version": REPORT_VERSION,
        "heatmaps": {
            "path": "routing_heatmaps.npy",
            "shape": list(assignments.shape),
            "utilization": heatmap_utilization(assignments, len(routings[0].mask[0, 0])).tolist(),
        },
        "specialization": specialization,
        "knockout": knockout,
        "composition": composition,
        "dynamics": dynamics,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report
