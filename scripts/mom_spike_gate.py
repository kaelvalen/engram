"""Spike-gate evaluation (spec §6.4) from finished run logs.

Reads output/mom/mom-spike-{mom,B1,B2,B3}/seed*.json and reports, per model
and context: mean ± std recall over seeds, final min expert utilization, and
the gate verdict:

    PASS  if mom-spike ≥ B1 AND ≥ max(B2, B3) on recall @ 4k context
          AND final min expert utilization ≥ 10% (no hard collapse).

Usage:
    python scripts/mom_spike_gate.py [--runs-dir output/mom] [--gate-context 4096]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

KINDS = ("mom", "B1", "B2", "B3")


def _collect(runs_dir: Path) -> dict:
    data = {}
    for kind in KINDS:
        seeds = sorted((runs_dir / f"mom-spike-{kind}").glob("seed*.json"))
        if not seeds:
            # also accept the direct experiment dir (mom-spike for kind=mom)
            alt = (runs_dir / ("mom-spike" if kind == "mom" else f"mom-spike-{kind}")).glob(
                "seed*.json"
            )
            seeds = sorted(alt)
        if seeds:
            data[kind] = [json.loads(p.read_text()) for p in seeds]
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="output/mom")
    ap.add_argument("--gate-context", type=int, default=4096)
    args = ap.parse_args()

    data = _collect(Path(args.runs_dir))
    missing = [k for k in KINDS if k not in data]
    if missing:
        print(f"WARNING: missing runs for {missing}")

    summary = {}
    contexts = set()
    for kind, runs in data.items():
        by_ctx = {}
        for r in runs:
            for ctx, acc in r.get("accuracy_by_context", {}).items():
                by_ctx.setdefault(int(ctx), []).append(acc)
                contexts.add(int(ctx))
        final_acc = [r["final"].get("accuracy", 0.0) for r in runs]
        min_utils = [
            r["final"]["min_utilization"]
            for r in runs
            if r["final"].get("min_utilization") is not None
        ]
        summary[kind] = {
            "by_ctx": by_ctx,
            "final_acc": final_acc,
            "min_util": min_utils,
            "params": runs[0].get("param_report", {}).get("total"),
        }

    ctx_sorted = sorted(contexts)
    header = (
        "model  " + "  ".join(f"recall@{c}" for c in ctx_sorted) + "  final@train  min_util  params"
    )
    print(header)
    print("-" * len(header))
    for kind in KINDS:
        if kind not in summary:
            continue
        s = summary[kind]
        cells = []
        for c in ctx_sorted:
            accs = s["by_ctx"].get(c)
            cells.append(
                f"{torch.tensor(accs).mean():.3f}±{torch.tensor(accs).std():.3f}"
                if accs
                else "  —  "
            )
        fa = torch.tensor(s["final_acc"])
        mu = f"{min(s['min_util']):.3f}" if s["min_util"] else " — "
        print(
            f"{kind:<5}  "
            + "  ".join(cells)
            + f"  {fa.mean():.3f}±{fa.std():.3f}  {mu:>7}  {s['params']}"
        )

    # ---- gate verdict ----
    if "mom" not in summary or not any(k in summary for k in ("B1", "B2", "B3")):
        print("\nGATE: INCONCLUSIVE (missing mom or baselines)")
        return 2
    gate_ctx = args.gate_context
    mom_acc = torch.tensor(summary["mom"]["by_ctx"].get(gate_ctx, [0.0])).mean().item()
    rivals = {}
    for k in ("B1", "B2", "B3"):
        if k in summary:
            rivals[k] = torch.tensor(summary[k]["by_ctx"].get(gate_ctx, [0.0])).mean().item()
    min_util = min(summary["mom"]["min_util"]) if summary["mom"]["min_util"] else 0.0

    ok_quality = all(mom_acc >= v for v in rivals.values()) and rivals
    ok_collapse = min_util >= 0.10
    print(
        f"\nrecall@{gate_ctx}: mom={mom_acc:.3f} | "
        + "  ".join(f"{k}={v:.3f}" for k, v in rivals.items())
    )
    print(f"min expert utilization: {min_util:.3f} (need ≥ 0.10)")
    verdict = (
        "PASS"
        if (ok_quality and ok_collapse)
        else ("FAIL (collapse)" if not ok_collapse else "FAIL (quality)")
    )
    print(f"SPIKE GATE: {verdict}")
    return 0 if (ok_quality and ok_collapse) else 1


if __name__ == "__main__":
    raise SystemExit(main())
