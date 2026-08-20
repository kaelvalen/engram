"""Stage-0 fixed-scale surprise probe (experiment design §6.4).

Sweeps a *frozen* per-expert surprise_weight (and scale) while the per-layer
SurprisePredictor trains, to ask the cheapest question first: "is the learned
surprise signal useful AT ALL?" (before any learnability work). Not
architectural learning — a fixed-scale insurance probe. If no setting moves
recall, the predictor/signal design is suspect before Stage 1/2.

Each grid cell freezes surprise_weight (probe) and trains the predictor via its
aux MSE loss. The same-sign cell `w[1,1]` is a shift-invariant control (equal
shift over all experts => no routing effect, by softmax/argmax shift-invariance).

Usage (real numbers on the RTX 5060 / GPU):
    python scripts/mom_surprise_probe.py --config configs/mom/surprise_probe.yaml \
        --seeds 0,1,2 --device cuda
CPU validation (tiny model, short steps):
    python scripts/mom_surprise_probe.py --config <small> --seeds 0 \
        --steps 120 --device cpu --log-dir /tmp/mom_probe
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml
from mom.train import train_one

# (tag, model-field overrides). Every non-"off" cell trains the predictor and
# freezes the swept per-expert surprise_weight.
GRID = [
    ("off", {"use_surprise_predictor": False, "router_surprise_scale": 0.0}),
    (
        "w[1,-1]",
        {
            "use_surprise_predictor": True,
            "router_surprise_scale": 1.0,
            "router_surprise_weight": (1.0, -1.0),
            "freeze_surprise_weight": True,
        },
    ),
    (
        "w[-1,1]",
        {
            "use_surprise_predictor": True,
            "router_surprise_scale": 1.0,
            "router_surprise_weight": (-1.0, 1.0),
            "freeze_surprise_weight": True,
        },
    ),
    (
        "w[1,0]",
        {
            "use_surprise_predictor": True,
            "router_surprise_scale": 1.0,
            "router_surprise_weight": (1.0, 0.0),
            "freeze_surprise_weight": True,
        },
    ),
    (
        "w[0,1]",
        {
            "use_surprise_predictor": True,
            "router_surprise_scale": 1.0,
            "router_surprise_weight": (0.0, 1.0),
            "freeze_surprise_weight": True,
        },
    ),
    (
        "w[1,1]",
        {
            "use_surprise_predictor": True,
            "router_surprise_scale": 1.0,
            "router_surprise_weight": (1.0, 1.0),
            "freeze_surprise_weight": True,
        },
    ),  # same-sign control: expected no-op (shift-invariant)
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", default="0", help="comma-separated seed list")
    ap.add_argument("--steps", type=int, default=None, help="override optim.steps (CPU validation)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log-dir", default="output/mom")
    args = ap.parse_args()

    with open(args.config) as f:
        base = yaml.safe_load(f)
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.steps is not None:
        base["optim"]["steps"] = args.steps

    results = {}
    for tag, ov in GRID:
        cfg = copy.deepcopy(base)
        cfg["model"].update(ov)
        cfg["experiment"] = f"mom-surprise-{tag}"
        runs = [train_one(cfg, seed, device=args.device) for seed in seeds]
        results[tag] = runs
        # row summary: mean ± std recall per context
        contexts = sorted({int(c) for r in runs for c in r.get("accuracy_by_context", {})})
        row = []
        for ctx in contexts:
            accs = [r["accuracy_by_context"].get(str(ctx), float("nan")) for r in runs]
            t = torch.tensor(accs, dtype=torch.float)
            s = f"@{ctx}={t.mean():.3f}" + (f"±{t.std():.3f}" if t.numel() > 1 else "")
            row.append(s.replace("nan", " - "))
        print(f"  [{tag}] " + " ".join(row))

    # persist a compact summary next to the cell dirs
    out = Path(args.log_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "mom-surprise-probe.json").write_text(
        json.dumps(
            {
                tag: [
                    {
                        "seed": r["seed"],
                        "final": r["final"],
                        "accuracy_by_context": r.get("accuracy_by_context", {}),
                    }
                    for r in runs
                ]
                for tag, runs in results.items()
            },
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
