"""Aggregate per-seed benchmark checkpoints into mean ± std tables.

Walks <results>/<config>/seed<N>/best_*.pt, reads the recorded val metrics, and
prints a markdown table grouped by config. Report mean ± std across seeds —
never single-best.

    python scripts/aggregate_results.py output/benchmarks
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_dir", type=str)
    p.add_argument("--metric", default="val_acc", help="metric key in ckpt['metrics']")
    args = p.parse_args()

    root = Path(args.results_dir)
    by_config: dict[str, list[float]] = defaultdict(list)

    for ckpt_path in sorted(root.glob("*/seed*/best_*.pt")):
        config = ckpt_path.parent.parent.name
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            metrics = ckpt.get("metrics", {})
            if args.metric in metrics:
                by_config[config].append(float(metrics[args.metric]))
        except Exception as e:  # noqa: BLE001
            print(f"! skip {ckpt_path}: {e}")

    if not by_config:
        print(f"No checkpoints with metric '{args.metric}' under {root}/")
        return

    print(f"\n| Config | seeds | {args.metric} (mean ± std) |")
    print("|---|---|---|")
    for config in sorted(by_config):
        vals = by_config[config]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"| {config} | {len(vals)} | {mean:.4f} ± {std:.4f} |")


if __name__ == "__main__":
    main()
