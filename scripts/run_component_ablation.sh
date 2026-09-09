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

  # ----- Component 4: Matrix Key-Value Memory vs Vector SSM State -----
  run "ssd_only"          "$SEED" --ssm-kind ssd --block-pattern s4
  run "delta_only"        "$SEED" --block-pattern delta

  # ----- Component 1: Local Context (LocalConv) -----
  run "no_conv"           "$SEED" --ssm-kind ssd --conv-kernel-size 0

  # ----- Component 2: Multiplicative Output Gating (GatedFusion) -----
  run "no_out_gate"       "$SEED" --ssm-kind ssd --no-delta-out-gate

  # ----- Component 3: Recurrent State Memory (RecurrentState vs Memoryless) -----
  run "delta_memoryless"  "$SEED" --block-pattern delta --delta-memoryless

  # ----- Component 5: Token-wise Capacity (SwiGLU FFN) -----
  run "no_ffn"            "$SEED" --ssm-kind ssd --ffn-expand 0

  # ----- Combined Ablation -----
  run "pure_recurrent"    "$SEED" --ssm-kind ssd --conv-kernel-size 0 --ffn-expand 0

  # ----- Pooling strategy -----
  run "pool_last"         "$SEED" --ssm-kind ssd --pool-type last
done

echo ""
echo "Done. Aggregate with:"
echo "  python scripts/aggregate_results.py $RESULTS --metric val_macro_auc --detailed"
echo "  python scripts/statistical_analysis.py $RESULTS --metric val_macro_auc --output $RESULTS/stats_report.json"
