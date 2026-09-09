"""Aggregate per-seed benchmark checkpoints into mean ± std tables.

Walks <results>/<config>/seed<N>/best_*.pt, reads the recorded val metrics, and
prints a markdown table grouped by config. Report mean ± std across seeds -
never single-best.

Enhanced (2026-09-09): paired t-test, Cohen's d, bootstrap CI, parameter
efficiency (AUROC / 100k params), and detailed JSON export.

    python scripts/aggregate_results.py output/benchmarks
    python scripts/aggregate_results.py output/benchmarks --metric val_macro_auc --detailed
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) from a percentile bootstrap on `values`."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if n == 1:
        return values[0], values[0], values[0]
    means = sorted(statistics.mean(rng.choices(values, k=n)) for _ in range(n_resamples))
    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q
    lo = means[max(0, int(lo_q * len(means)))]
    hi = means[min(len(means) - 1, int(hi_q * len(means)))]
    return statistics.mean(values), lo, hi


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d for paired samples."""
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    diffs = [ai - bi for ai, bi in zip(a, b)]
    mean_d = statistics.mean(diffs)
    std_d = statistics.stdev(diffs)
    if std_d == 0:
        return float("inf") if mean_d != 0 else 0.0
    return mean_d / std_d


def hedges_g(a: list[float], b: list[float]) -> float:
    """Hedges' g: small-sample bias corrected paired effect size."""
    d = cohens_d(a, b)
    n = len(a)
    if math.isnan(d) or n < 2:
        return float("nan")
    df = n - 1
    if 4 * df <= 1:
        return d
    j = 1.0 - (3.0 / (4.0 * df - 1.0))
    return d * j


def classify_evidence_tier(
    delta_mean: float,
    g_val: float,
    p_val: float,
    ci_lo: float,
    ci_hi: float,
    mde: float = float("nan"),
) -> tuple[int, str]:
    """Classify comparison into 3 Evidence Tiers:
    Tier 1 (Significant & Substantial):
        p < 0.05 and |g| >= 0.5 and CI does not span 0.
    Tier 2 (Directional Trend / Inconclusive):
        p >= 0.05 or CI spans 0, but |g| >= 0.2.
    Tier 3 (Measurement Noise / Practical Equivalence):
        |g| < 0.2 or |delta| <= MDE.
    """
    if not math.isnan(p_val) and p_val < 0.05 and not math.isnan(g_val) and abs(g_val) >= 0.5:
        if (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0):
            return 1, "Tier 1: Significant & Substantial"
    if not math.isnan(g_val) and abs(g_val) < 0.2:
        return 3, "Tier 3: Measurement Noise / Equivalence"
    if not math.isnan(mde) and abs(delta_mean) <= mde:
        return 3, "Tier 3: Measurement Noise / Equivalence"
    return 2, "Tier 2: Directional Trend"


def paired_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
    """Paired t-test. Returns (t_stat, p_value).

    Falls back to a manual calculation if scipy is not available.
    """
    if len(a) != len(b) or len(a) < 2:
        return float("nan"), float("nan")
    try:
        from scipy.stats import ttest_rel

        res = ttest_rel(a, b)
        return float(res.statistic), float(res.pvalue)
    except ImportError:
        pass

    # Manual paired t-test
    n = len(a)
    diffs = [ai - bi for ai, bi in zip(a, b)]
    mean_d = statistics.mean(diffs)
    std_d = statistics.stdev(diffs)
    if std_d == 0:
        return float("inf") if mean_d != 0 else 0.0, 0.0 if mean_d != 0 else 1.0
    t_stat = mean_d / (std_d / math.sqrt(n))
    # Two-tailed p-value approximation using the t-distribution
    # For small n, this is rough but better than nothing without scipy.
    df = n - 1
    try:
        from scipy.stats import t as t_dist

        p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df))
    except ImportError:
        # Very rough approximation for df >= 2
        # Use the normal approximation for large t
        p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
    return t_stat, p_value


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF (Abramowitz & Stegun)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def minimum_detectable_effect(
    n_seeds: int, std: float, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Approximate MDE for a paired t-test (two-tailed)."""
    if n_seeds < 2 or std <= 0:
        return float("nan")
    # z_alpha/2 ≈ 1.96 for alpha=0.05, z_beta ≈ 0.84 for power=0.8
    z_a = 1.96 if alpha == 0.05 else 2.576
    z_b = 0.84 if power == 0.8 else 1.28
    return (z_a + z_b) * std / math.sqrt(n_seeds)


# ---------------------------------------------------------------------------
# Checkpoint scanning
# ---------------------------------------------------------------------------


def scan_checkpoints(
    root: Path, metric: str
) -> tuple[dict[str, list[float]], dict[str, int | None]]:
    """Scan checkpoints and return per-config metric values and param counts."""
    by_config: dict[str, list[float]] = defaultdict(list)
    param_counts: dict[str, int | None] = {}

    for ckpt_path in sorted(root.glob("*/seed*/best_*.pt")):
        config = ckpt_path.parent.parent.name
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            metrics = ckpt.get("metrics", {})
            if metric in metrics:
                by_config[config].append(float(metrics[metric]))

            # Try to extract parameter count from checkpoint
            if config not in param_counts:
                model_state = ckpt.get("model_state", ckpt.get("model", {}))
                if model_state:
                    total_params = sum(
                        v.numel() for v in model_state.values() if isinstance(v, torch.Tensor)
                    )
                    param_counts[config] = total_params if total_params > 0 else None
                else:
                    param_counts[config] = None
        except Exception as e:  # noqa: BLE001
            print(f"! skip {ckpt_path}: {e}")

    return dict(by_config), param_counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_dir", type=str)
    p.add_argument("--metric", default="val_acc", help="metric key in ckpt['metrics']")
    p.add_argument(
        "--detailed",
        action="store_true",
        help="Print pairwise t-tests, effect sizes, bootstrap CIs",
    )
    p.add_argument(
        "--json",
        type=str,
        default=None,
        help="Write detailed results to this JSON file",
    )
    p.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Config name to use as the reference for pairwise tests (default: best mean)",
    )
    args = p.parse_args()

    root = Path(args.results_dir)
    by_config, param_counts = scan_checkpoints(root, args.metric)

    if not by_config:
        print(f"No checkpoints with metric '{args.metric}' under {root}/")
        return

    # ---- Summary table ----
    print(f"\n## Summary - {args.metric}\n")
    print(
        f"| Config | Seeds | {args.metric} (mean ± std) | Seed-level 95% CI | Params | {args.metric}/100k |"
    )
    print("|---|---|---|---|---|---|")

    results_json: dict = {"metric": args.metric, "configs": {}}

    for config in sorted(by_config):
        vals = by_config[config]
        n = len(vals)
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if n > 1 else 0.0

        _, ci_lo, ci_hi = bootstrap_ci(vals)
        ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]" if n > 1 else "-"

        params = param_counts.get(config)
        params_str = f"~{params // 1000}k" if params else "?"
        efficiency = mean / (params / 100_000) if params else float("nan")
        eff_str = f"{efficiency:.4f}" if not math.isnan(efficiency) else "?"

        mde = minimum_detectable_effect(n, std)

        print(f"| {config} | {n} | {mean:.4f} ± {std:.4f} | {ci_str} | {params_str} | {eff_str} |")

        results_json["configs"][config] = {
            "n_seeds": n,
            "values": vals,
            "mean": mean,
            "std": std,
            "ci_95": [ci_lo, ci_hi],
            "params": params,
            "efficiency_per_100k": efficiency,
            "mde_at_power_0.8": mde,
        }

    # ---- Pairwise tests (detailed mode) ----
    if args.detailed and len(by_config) > 1:
        configs = sorted(by_config.keys())

        # Find reference config (best mean or user-specified)
        if args.reference and args.reference in by_config:
            ref = args.reference
        else:
            ref = max(configs, key=lambda c: statistics.mean(by_config[c]))

        ref_vals = by_config[ref]

        print(f"\n## Pairwise tests vs reference: `{ref}`\n")
        print("| Config | Δ mean | Hedges' g (Cohen's d) | t-stat | p-value | Evidence Tier |")
        print("|---|---|---|---|---|---|")

        pairwise_json: list[dict] = []
        for config in configs:
            if config == ref:
                continue
            vals = by_config[config]
            # Align seeds: use min(len) pairs
            n_pairs = min(len(ref_vals), len(vals))
            if n_pairs < 2:
                print(f"| {config} | - | - | - | - | too few seeds |")
                continue

            a, b = ref_vals[:n_pairs], vals[:n_pairs]
            delta = statistics.mean(a) - statistics.mean(b)
            d = cohens_d(a, b)
            g = hedges_g(a, b)
            t_stat, p_val = paired_ttest(a, b)

            # Pairwise difference bootstrap CI
            diffs = [ai - bi for ai, bi in zip(a, b)]
            _, diff_ci_lo, diff_ci_hi = bootstrap_ci(diffs)

            std_ref = statistics.stdev(ref_vals) if len(ref_vals) > 1 else 0.0
            mde = minimum_detectable_effect(n_pairs, std_ref)

            tier_num, tier_label = classify_evidence_tier(
                delta, g, p_val, diff_ci_lo, diff_ci_hi, mde
            )

            print(
                f"| {config} | {delta:+.4f} | {g:.2f} ({d:.2f}) | {t_stat:.3f} | {p_val:.4f} | **{tier_label}** |"
            )

            pairwise_json.append(
                {
                    "config": config,
                    "vs": ref,
                    "delta_mean": delta,
                    "cohens_d": d,
                    "hedges_g": g,
                    "t_stat": t_stat,
                    "p_value": p_val,
                    "evidence_tier": tier_num,
                    "evidence_label": tier_label,
                    "diff_ci_95": [diff_ci_lo, diff_ci_hi],
                    "n_pairs": n_pairs,
                }
            )

        results_json["pairwise_tests"] = pairwise_json
        results_json["reference"] = ref

        # ---- Power analysis ----
        print("\n## Power analysis (α=0.05, power=0.80)\n")
        print("| Config | seeds | pooled std | MDE |")
        print("|---|---|---|---|")
        for config in configs:
            vals = by_config[config]
            n = len(vals)
            std = statistics.stdev(vals) if n > 1 else 0.0
            mde = minimum_detectable_effect(n, std)
            mde_str = f"{mde:.4f}" if not math.isnan(mde) else "-"
            print(f"| {config} | {n} | {std:.4f} | {mde_str} |")

    # ---- JSON export ----
    if args.json:
        Path(args.json).write_text(json.dumps(results_json, indent=2, default=str))
        print(f"\nDetailed results written to {args.json}")


if __name__ == "__main__":
    main()
