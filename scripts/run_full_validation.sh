#!/usr/bin/env bash
# Master validation script: runs ALL experiments and produces statistical reports.
#
# This is the single command to reproduce every claim in the paper on the local
# laptop GPU. It runs:
#   1. ECG main table: 4 ENGRAM variants + 2 baselines × 10 seeds
#   2. Vision: engram_image × 5 seeds
#   3. Audio: engram_audio + audio_cnn baseline × 5 seeds
#   4. Component ablation × 5 seeds
#   5. Statistical analysis on all results
#
# Estimated runtime on RTX 5060 (8 GB): 10-14 hours.
#
# Usage:
#   DATA_ROOT=./datasets bash scripts/run_full_validation.sh
set -euo pipefail

if [[ -z "${TRITON_LIBCUDA_PATH:-}" && -e /run/opengl-driver/lib/libcuda.so.1 ]]; then
  export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
fi

DATA_ROOT="${DATA_ROOT:-./datasets}"
ECG_SEEDS="${ECG_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
CROSS_SEEDS="${CROSS_SEEDS:-0 1 2 3 4}"
ABLATION_SEEDS="${ABLATION_SEEDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-10}"
EARLY_STOPPING="${EARLY_STOPPING:-3}"
AMP="${AMP:-bf16}"
RESULTS="${RESULTS:-output/full_validation}"
mkdir -p "$RESULTS"

COMMON="--hidden-dim 64 --num-layers 4 --num-heads 4 --batch-size 8 --amp $AMP --data-root $DATA_ROOT --epochs $EPOCHS --early-stopping $EARLY_STOPPING"

run () {
  local name="$1"; local seed="$2"; shift 2
  local extra=""
  case " $* " in
    *" --modality ecg "*) extra="--ecg-multilabel";;
    *" --modality audio "*) extra="--no-audio-synthetic";;
  esac
  echo ">>> [$name | seed=$seed] $* $extra"
  PYTHONHASHSEED="$seed" engram-train $COMMON --seed "$seed" --output-dir "$RESULTS/$name/seed$seed" "$@" $extra \
    2>&1 | tee "$RESULTS/${name}_seed${seed}.log"
}

echo "============================================"
echo "  ENGRAM Full Validation Pipeline"
echo "  ECG seeds: $ECG_SEEDS"
echo "  Cross-modal seeds: $CROSS_SEEDS"
echo "  Ablation seeds: $ABLATION_SEEDS"
echo "  Epochs: $EPOCHS"
echo "============================================"

# ---- Phase 1: ECG main table (10 seeds) ----
echo ""
echo "=== Phase 1/5: ECG main table ==="
for SEED in $ECG_SEEDS; do
  run "engram_hybrid_ecg"    "$SEED" --modality ecg --ssm-kind ssd
  run "mamba2_only_ecg"      "$SEED" --modality ecg --ssm-kind ssd --block-pattern s4
  run "gateddelta_only_ecg"  "$SEED" --modality ecg --block-pattern delta
  run "engram_legacy_ecg"    "$SEED" --modality ecg --ssm-kind s4d_legacy --s4d-init lin

  python scripts/train_baseline.py --model resnet1d --task ecg --epochs "$EPOCHS" --seed "$SEED" \
    --window-size 1000 --ecg-task superdiag --ecg-multilabel --data-root "$DATA_ROOT" --batch-size 8 \
    --output-dir "$RESULTS/resnet1d_ecg/seed$SEED" 2>&1 | tee "$RESULTS/resnet1d_ecg_seed${SEED}.log"
  python scripts/train_baseline.py --model transformer --task ecg --epochs "$EPOCHS" --seed "$SEED" \
    --window-size 1000 --ecg-task superdiag --ecg-multilabel --data-root "$DATA_ROOT" --batch-size 8 \
    --output-dir "$RESULTS/transformer_ecg/seed$SEED" 2>&1 | tee "$RESULTS/transformer_ecg_seed${SEED}.log"
done

# ---- Phase 2: Vision (5 seeds) ----
echo ""
echo "=== Phase 2/5: Vision ==="
for SEED in $CROSS_SEEDS; do
  run "engram_image" "$SEED" --modality image
done

# ---- Phase 3: Audio (5 seeds) ----
echo ""
echo "=== Phase 3/5: Audio ==="
for SEED in $CROSS_SEEDS; do
  run "engram_audio" "$SEED" --modality audio

  python scripts/train_baseline.py --model audio_cnn --task audio --epochs "$EPOCHS" --seed "$SEED" \
    --data-root "$DATA_ROOT" --batch-size 8 \
    --output-dir "$RESULTS/audio_cnn/seed$SEED" 2>&1 | tee "$RESULTS/audio_cnn_seed${SEED}.log"
done

# ---- Phase 4: Component ablation (5 seeds) ----
echo ""
echo "=== Phase 4/5: Component ablation ==="
SEEDS="$ABLATION_SEEDS" EPOCHS="$EPOCHS" DATA_ROOT="$DATA_ROOT" AMP="$AMP" \
  RESULTS="$RESULTS/ablation" bash scripts/run_component_ablation.sh

# ---- Phase 5: Statistical analysis ----
echo ""
echo "=== Phase 5/5: Statistical analysis ==="

echo "--- ECG statistical analysis ---"
python scripts/aggregate_results.py "$RESULTS" --metric val_macro_auc --detailed \
  --json "$RESULTS/ecg_aggregate.json"
python scripts/statistical_analysis.py "$RESULTS" --metric val_macro_auc \
  --output "$RESULTS/ecg_stats_report.json"

echo "--- Vision aggregate ---"
python scripts/aggregate_results.py "$RESULTS" --metric val_acc \
  --json "$RESULTS/vision_aggregate.json"

echo "--- Ablation statistical analysis ---"
python scripts/aggregate_results.py "$RESULTS/ablation" --metric val_macro_auc --detailed \
  --json "$RESULTS/ablation_aggregate.json"
python scripts/statistical_analysis.py "$RESULTS/ablation" --metric val_macro_auc \
  --output "$RESULTS/ablation_stats_report.json"

echo ""
echo "============================================"
echo "  All experiments complete!"
echo "  Results: $RESULTS/"
echo "  ECG report: $RESULTS/ecg_stats_report.json"
echo "  Ablation report: $RESULTS/ablation_stats_report.json"
echo "============================================"
