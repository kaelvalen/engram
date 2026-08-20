"""Generate the MoM §7 analysis report from a finished spike run.

Loads a checkpoint written by mom.train (seed{N}.pt) plus the run's
metrics json (seed{N}.json) and produces the versioned analysis report
(routing heatmaps, specialization MI, expert knockout, learned composition
vs the 3:1 line, dynamics) under an output directory.

Usage:
    python scripts/mom_analysis.py output/mom/mom-spike-mom/seed0.pt \
        output/mom/mom-spike-mom/seed0.json output/mom/analysis/seed0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mom.analysis import generate_report
from mom.baselines import build_model
from mom.config import MoMConfig
from mom.tasks.mqar import MQARConfig, make_mqar_batch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="seed{N}.pt written by mom.train")
    ap.add_argument("metrics", help="seed{N}.json with the run history")
    ap.add_argument("out_dir")
    args = ap.parse_args()

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = blob["config"]
    metrics = json.loads(Path(args.metrics).read_text())

    cfg = MoMConfig.from_dict(config["model"])
    model = build_model(config["model"].get("kind", "mom"), cfg, config["task"]["vocab_size"])
    model.load_state_dict(blob["model_state"])
    model.eval()

    task_cfg = MQARConfig(**{k: v for k, v in config["task"].items() if k != "name"})
    ids, labels, classes = make_mqar_batch(
        task_cfg, 8, torch.Generator().manual_seed(999), return_classes=True
    )

    report = generate_report(
        model,
        ids,
        labels,
        metrics["history"],
        args.out_dir,
        token_classes=classes,
        experts=tuple(cfg.experts),
    )
    out = Path(args.out_dir) / "report.json"
    print(f"analysis report → {out}")
    print("  heatmap util:", report["heatmaps"]["utilization"])
    print("  knockout:", json.dumps(report["knockout"]))
    if report["specialization"]:
        for i, s in enumerate(report["specialization"]["per_layer"]):
            print(f"  layer{i} MI={s['mi']:.3f} p={s['p_value']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
