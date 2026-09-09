#!/usr/bin/env bash
# Component ablation experiments for ENGRAM on PTB-XL ECG.
#
# Tests which components (conv, FFN, pooling strategy) actually contribute to
# performance. This is the "leave-one-out" ablation that reviewers expect.
#
# All runs use the laptop-friendly reduced config (hidden_dim=64, num_layers=4)
# and the same multi-label macro-AUROC protocol as the main benchmark.
#
# Usage:
#   DATA_ROOT=./datasets SEEDS="0 1 2 3 4" EPOCHS=10 bash scripts/run_component_ablation.sh
set -euo pipefail

if [[ -z "${TRITON_LIBCUDA_PATH:-}" && -e /run/opengl-driver/lib/libcuda.so.1 ]]; then
  export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
fi

DATA_ROOT="${DATA_ROOT:-./datasets}"
SEEDS="${SEEDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-10}"
EARLY_STOPPING="${EARLY_STOPPING:-3}"
AMP="${AMP:-bf16}"
RESULTS="${RESULTS:-output/ablation_components}"
mkdir -p "$RESULTS"

COMMON="--hidden-dim 64 --num-layers 4 --num-heads 4 --batch-size 8 --amp $AMP --data-root $DATA_ROOT --epochs $EPOCHS --early-stopping $EARLY_STOPPING"

run () {
  local name="$1"; local seed="$2"; shift 2
  echo ">>> [$name | seed=$seed] $*"
  PYTHONHASHSEED="$seed" engram-train $COMMON --seed "$seed" --output-dir "$RESULTS/$name/seed$seed" \
    --modality ecg --ecg-multilabel "$@" \
    2>&1 | tee "$RESULTS/${name}_seed${seed}.log"
}

for SEED in $SEEDS; do
  # ----- Control: full hybrid model -----
  run "full_hybrid"       "$SEED" --ssm-kind ssd

  # ----- Architecture ablations (already in main benchmark, but repeated here for consistency) -----
  run "ssd_only"          "$SEED" --ssm-kind ssd --block-pattern s4
  run "delta_only"        "$SEED" --block-pattern delta

  # ----- Component ablations: remove one component at a time -----

  # No short causal conv (tests if local mixing helps the SSM)
  run "no_conv"           "$SEED" --ssm-kind ssd --conv-kernel-size 0

  # No SwiGLU FFN (tests if per-position nonlinear expansion matters)
  run "no_ffn"            "$SEED" --ssm-kind ssd --ffn-expand 0

  # No conv AND no FFN (pure mixer backbone, maximally ablated)
  run "no_conv_no_ffn"    "$SEED" --ssm-kind ssd --conv-kernel-size 0 --ffn-expand 0

  # Pooling: last token instead of mean
  run "pool_last"         "$SEED" --ssm-kind ssd --pool-type last

  # ----- Dropout ablation -----
  run "dropout_0.1"       "$SEED" --ssm-kind ssd --dropout 0.1
  run "dropout_0.2"       "$SEED" --ssm-kind ssd --dropout 0.2
done

echo ""
echo "Done. Aggregate with:"
echo "  python scripts/aggregate_results.py $RESULTS --metric val_macro_auc --detailed"
echo "  python scripts/statistical_analysis.py $RESULTS --metric val_macro_auc --output $RESULTS/stats_report.json"
