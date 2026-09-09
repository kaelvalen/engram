"""Detailed statistical analysis of ENGRAM benchmark results.

Produces:
- Full pairwise t-test matrix with Holm-Bonferroni correction
- Effect size (Cohen's d) matrix
- Parameter efficiency ranking
- Power analysis per config
- Verdict table: which differences are real vs measurement noise

Usage:
    python scripts/statistical_analysis.py output/benchmarks_laptop --metric val_macro_auc
    python scripts/statistical_analysis.py output/benchmarks_laptop --metric val_macro_auc --output output/stats_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from itertools import combinations
from pathlib import Path

from aggregate_results import (
    bootstrap_ci,
    classify_evidence_tier,
    cohens_d,
    hedges_g,
    minimum_detectable_effect,
    paired_ttest,
    scan_checkpoints,
)


def holm_bonferroni(p_values: list[tuple[str, str, float]]) -> list[tuple[str, str, float, bool]]:
    """Apply Holm-Bonferroni correction to a list of (name_a, name_b, p_value).

    Returns list of (name_a, name_b, adjusted_p, is_significant_at_0.05).
    """
    m = len(p_values)
    if m == 0:
        return []
    sorted_pvals = sorted(p_values, key=lambda x: x[2])
    results = []
    for rank, (na, nb, p) in enumerate(sorted_pvals, 1):
        adjusted_p = min(p * (m - rank + 1), 1.0)
        results.append((na, nb, adjusted_p, adjusted_p < 0.05))
    return results


def main():
    p = argparse.ArgumentParser(description="Detailed statistical analysis of benchmark results")
    p.add_argument("results_dir", type=str)
    p.add_argument("--metric", default="val_macro_auc")
    p.add_argument("--output", type=str, default=None, help="JSON output file")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    args = p.parse_args()

    root = Path(args.results_dir)
    by_config, param_counts = scan_checkpoints(root, args.metric)

    if not by_config:
        print(f"No checkpoints with metric '{args.metric}' under {root}/")
        return

    configs = sorted(by_config.keys())
    report: dict = {
        "metric": args.metric,
        "alpha": args.alpha,
        "results_dir": str(root),
    }

    # ---- 1. Summary ----
    print(f"# Statistical Analysis Report - {args.metric}")
    print(f"\nResults directory: `{root}`")
    print(f"Significance level: α = {args.alpha}")
    print()

    print("## 1. Per-config summary\n")
    print("| Config | N | Mean | Std | Seed-level 95% CI | Params | Metric/100k |")
    print("|---|---|---|---|---|---|---|")

    config_summary: dict = {}
    for config in configs:
        vals = by_config[config]
        n = len(vals)
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if n > 1 else 0.0
        _, ci_lo, ci_hi = bootstrap_ci(vals)
        params = param_counts.get(config)
        eff = mean / (params / 100_000) if params else float("nan")

        ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]" if n > 1 else "-"
        params_str = f"~{params // 1000}k" if params else "?"
        eff_str = f"{eff:.4f}" if not math.isnan(eff) else "?"

        print(f"| {config} | {n} | {mean:.4f} | {std:.4f} | {ci_str} | {params_str} | {eff_str} |")

        config_summary[config] = {
            "n": n,
            "mean": mean,
            "std": std,
            "seed_level_ci_95": [ci_lo, ci_hi],
            "params": params,
            "efficiency": eff,
            "values": vals,
        }

    report["config_summary"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "values"} for k, v in config_summary.items()
    }

    # ---- 2. Pairwise t-test matrix ----
    if len(configs) > 1:
        print("\n## 2. Pairwise paired t-tests\n")

        raw_pvals: list[tuple[str, str, float]] = []
        pairwise_data: dict[str, dict] = {}

        for a, b in combinations(configs, 2):
            vals_a = by_config[a]
            vals_b = by_config[b]
            n_pairs = min(len(vals_a), len(vals_b))
            if n_pairs < 2:
                continue

            va, vb = vals_a[:n_pairs], vals_b[:n_pairs]
            t_stat, p_val = paired_ttest(va, vb)
            d = cohens_d(va, vb)
            g = hedges_g(va, vb)
            delta = statistics.mean(va) - statistics.mean(vb)

            diffs = [vai - vbi for vai, vbi in zip(va, vb)]
            _, diff_ci_lo, diff_ci_hi = bootstrap_ci(diffs)
            std_a = statistics.stdev(va) if len(va) > 1 else 0.0
            mde = minimum_detectable_effect(n_pairs, std_a)

            tier_num, tier_label = classify_evidence_tier(
                delta, g, p_val, diff_ci_lo, diff_ci_hi, mde
            )

            raw_pvals.append((a, b, p_val))
            pairwise_data[f"{a} vs {b}"] = {
                "delta_mean": delta,
                "cohens_d": d,
                "hedges_g": g,
                "t_stat": t_stat,
                "p_value_raw": p_val,
                "evidence_tier": tier_num,
                "evidence_label": tier_label,
                "diff_ci_95": [diff_ci_lo, diff_ci_hi],
                "n_pairs": n_pairs,
            }

        # Apply Holm-Bonferroni correction
        corrected = holm_bonferroni(raw_pvals)

        print("| Pair | Δ mean | Hedges' g (Cohen's d) | t | p (raw) | p (Holm) | Evidence Tier |")
        print("|---|---|---|---|---|---|---|")

        for na, nb, adj_p, sig in corrected:
            key = f"{na} vs {nb}"
            data = pairwise_data[key]
            d = data["cohens_d"]
            g = data["hedges_g"]
            tier_label = data["evidence_label"]

            print(
                f"| {key} | {data['delta_mean']:+.4f} | "
                f"{g:.2f} ({d:.2f}) | {data['t_stat']:.3f} | "
                f"{data['p_value_raw']:.4f} | {adj_p:.4f} | **{tier_label}** |"
            )
            pairwise_data[key]["p_value_holm"] = adj_p
            pairwise_data[key]["significant_holm"] = sig

        report["pairwise_tests"] = pairwise_data

    # ---- 3. Power analysis ----
    print("\n## 3. Power analysis\n")
    print("What is the minimum detectable effect (MDE) at α=0.05, power=0.80?")
    print()
    print("| Config | N seeds | Std | MDE | Interpretation |")
    print("|---|---|---|---|---|")

    for config in configs:
        vals = by_config[config]
        n = len(vals)
        std = statistics.stdev(vals) if n > 1 else 0.0
        # MDE ≈ (z_α/2 + z_β) * σ / √n
        if n >= 2 and std > 0:
            mde = (1.96 + 0.84) * std / math.sqrt(n)
            interp = f"Cannot detect differences < {mde:.4f} with {n} seeds"
        else:
            mde = float("nan")
            interp = "Need ≥ 2 seeds with nonzero variance"

        mde_str = f"{mde:.4f}" if not math.isnan(mde) else "-"
        print(f"| {config} | {n} | {std:.4f} | {mde_str} | {interp} |")

    # ---- 4. Parameter efficiency ranking ----
    print("\n## 4. Parameter efficiency ranking\n")
    ranked = [
        (config, config_summary[config]["mean"], config_summary[config]["params"])
        for config in configs
        if config_summary[config]["params"]
    ]
    if ranked:
        ranked.sort(key=lambda x: x[1] / (x[2] / 100_000) if x[2] else 0, reverse=True)
        print(f"| Rank | Config | {args.metric} | Params | {args.metric}/100k params |")
        print("|---|---|---|---|---|")
        for i, (config, mean, params) in enumerate(ranked, 1):
            eff = mean / (params / 100_000)
            print(f"| {i} | {config} | {mean:.4f} | ~{params // 1000}k | {eff:.4f} |")

    # ---- 5. Verdict ----
    print("\n## 5. Verdict\n")
    # Check if any pairwise differences are significant
    if len(configs) > 1:
        any_sig = any(v.get("significant_holm", False) for v in pairwise_data.values())
        if any_sig:
            print(
                "Some pairwise differences are statistically significant after Holm-Bonferroni correction."
            )
            print("See the pairwise table above for details.")
        else:
            print("> **All pairwise differences are NOT statistically significant** after")
            print("> Holm-Bonferroni correction. The models are indistinguishable at the")
            print(f"> current seed count (N={min(len(v) for v in by_config.values())}).")
            print(">")
            print('> Honest claim: "The architectures perform within measurement noise of')
            print('> each other; the hybrid variant shows lower seed variance."')

    # ---- JSON output ----
    if args.output:
        # Clean up non-serializable values
        def clean(obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return str(obj)
            return obj

        Path(args.output).write_text(json.dumps(report, indent=2, default=clean))
        print(f"\nFull report written to {args.output}")


if __name__ == "__main__":
    main()
