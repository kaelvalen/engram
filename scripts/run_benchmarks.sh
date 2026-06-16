#!/usr/bin/env bash
# PRISM benchmark + ablation matrix (locked before running, per the paper plan).
#
# Reproduces every row of the paper tables. Datasets must be present under
# $DATA_ROOT (see EXPERIMENTS.md). Each config is run over $SEEDS seeds; report
# mean ± std, never single-best. Budget ≈ 30–60 single-GPU-hours.
#
# Usage:
#   DATA_ROOT=./datasets SEEDS="0 1 2" EPOCHS=50 bash scripts/run_benchmarks.sh
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./datasets}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-50}"
AMP="${AMP:-bf16}"
RESULTS="${RESULTS:-output/benchmarks}"
mkdir -p "$RESULTS"

# Shared backbone budget (~8M params).
COMMON="--hidden-dim 256 --num-layers 12 --num-heads 8 --amp $AMP --data-root $DATA_ROOT --epochs $EPOCHS"

run () {  # run <name> <seed> <extra-args...>
  local name="$1"; local seed="$2"; shift 2
  # PTB-XL is evaluated multi-label (macro AUROC) per the paper protocol.
  local extra=""
  case " $* " in *" --modality ecg "*) extra="--ecg-multilabel";; esac
  echo ">>> [$name | seed=$seed] $* $extra"
  PYTHONHASHSEED="$seed" prism-train $COMMON --output-dir "$RESULTS/$name/seed$seed" "$@" $extra \
    2>&1 | tee "$RESULTS/${name}_seed${seed}.log"
}

for SEED in $SEEDS; do
  # ---------------- Main table: architectures × modalities ----------------
  # PTB-XL super-diagnostic (primary), sCIFAR-10 (tertiary), Speech Commands (secondary).
  for MOD in ecg image audio; do
    run "prism_hybrid_$MOD"   "$SEED" --modality "$MOD" --ssm-kind ssd                          # SSD + Delta 3:1
    run "mamba2_only_$MOD"    "$SEED" --modality "$MOD" --ssm-kind ssd   --block-pattern s4      # SSD only
    run "gateddelta_only_$MOD" "$SEED" --modality "$MOD"                  --block-pattern delta   # Delta only
    run "prism_legacy_$MOD"   "$SEED" --modality "$MOD" --ssm-kind s4d_legacy --s4d-init lin      # S4D + Delta 3:1
  done
  # CNN / Transformer baselines (separate baseline trainer).
  python scripts/train_baseline.py --model resnet1d   --task ecg --epochs "$EPOCHS" \
    --output-dir "$RESULTS/resnet1d_ecg/seed$SEED" 2>&1 | tee "$RESULTS/resnet1d_ecg_seed${SEED}.log"
  python scripts/train_baseline.py --model transformer --task ecg --epochs "$EPOCHS" \
    --output-dir "$RESULTS/transformer_ecg/seed$SEED" 2>&1 | tee "$RESULTS/transformer_ecg_seed${SEED}.log"

  # ---------------- Ablations (on PTB-XL super-diag, cheapest) ----------------
  # 1) Layer pattern (matched 12 layers).
  run "abl_pattern_3to1"   "$SEED" --modality ecg --layer-pattern s4,s4,s4,delta,s4,s4,s4,delta,s4,s4,s4,delta
  run "abl_pattern_1to1"   "$SEED" --modality ecg --layer-pattern s4,delta,s4,delta,s4,delta,s4,delta,s4,delta,s4,delta
  run "abl_pattern_1to3"   "$SEED" --modality ecg --layer-pattern s4,delta,delta,delta,s4,delta,delta,delta,s4,delta,delta,delta
  run "abl_all_ssd"        "$SEED" --modality ecg --block-pattern s4
  run "abl_all_delta"      "$SEED" --modality ecg --block-pattern delta
  run "abl_delta_top"      "$SEED" --modality ecg --layer-pattern s4,s4,s4,s4,s4,s4,s4,s4,s4,delta,delta,delta
  run "abl_delta_bottom"   "$SEED" --modality ecg --layer-pattern delta,delta,delta,s4,s4,s4,s4,s4,s4,s4,s4,s4

  # 2) Depth at matched-ish budget (width compensates externally if desired).
  for L in 6 12 18 24; do run "abl_layers_$L" "$SEED" --modality ecg --num-layers "$L"; done

  # 3) Δ parameterisation: per-channel (SSD) vs per-head/mean-over-Dh (S4D legacy).
  run "abl_delta_perchannel" "$SEED" --modality ecg --ssm-kind ssd
  run "abl_delta_perhead"    "$SEED" --modality ecg --ssm-kind s4d_legacy

  # 4) Sliding-window attention every 4 layers (H1-style hybrid) on/off.
  run "abl_swa_on"  "$SEED" --modality ecg --layer-pattern s4,s4,s4,swa,s4,s4,s4,swa,s4,s4,s4,swa
  run "abl_swa_off" "$SEED" --modality ecg --ssm-kind ssd
done

echo "Done. Aggregate mean±std over seeds with: python scripts/aggregate_results.py $RESULTS"
