#!/usr/bin/env bash
# Laptop-friendly ENGRAM benchmark smoke (8 GB GPU).
#
# This is NOT the paper matrix: it keeps the same protocol (full 10 s ECG,
# multi-label macro-AUROC, mean±std over seeds) but uses a much smaller model
# and fewer epochs so it finishes on a single laptop GPU.
#
# Default runtime on an 8 GB laptop GPU is roughly 3–5 hours for EPOCHS=10
# and a single seed. Set EPOCHS=2 for a quick pipeline smoke (~40 min).
#
# Usage:
#   DATA_ROOT=./datasets SEEDS="0" EPOCHS=10 bash scripts/run_benchmarks_laptop.sh
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./datasets}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-10}"
AMP="${AMP:-bf16}"
RESULTS="${RESULTS:-output/benchmarks_laptop}"
RUN_ABLATIONS="${RUN_ABLATIONS:-0}"
mkdir -p "$RESULTS"

# Reduced backbone that fits into ~8 GB VRAM.
# (hidden_dim=64/num_layers=4 keeps the same ssd_state_dim=64 as the paper config
# while using a much smaller backbone; this is for pipeline validation only.)
COMMON="--hidden-dim 64 --num-layers 4 --num-heads 4 --batch-size 8 --amp $AMP --data-root $DATA_ROOT --epochs $EPOCHS"

run () {  # run <name> <seed> <extra-args...>
  local name="$1"; local seed="$2"; shift 2
  # PTB-XL is evaluated multi-label (macro AUROC) per the paper protocol.
  local extra=""
  case " $* " in *" --modality ecg "*) extra="--ecg-multilabel";; esac
  echo ">>> [$name | seed=$seed] $* $extra"
  PYTHONHASHSEED="$seed" engram-train $COMMON --seed "$seed" --output-dir "$RESULTS/$name/seed$seed" "$@" $extra \
    2>&1 | tee "$RESULTS/${name}_seed${seed}.log"
}

for SEED in $SEEDS; do
  # Main architectures on PTB-XL ECG.
  run "engram_hybrid_ecg"    "$SEED" --modality ecg --ssm-kind ssd
  run "mamba2_only_ecg"     "$SEED" --modality ecg --ssm-kind ssd --block-pattern s4
  run "gateddelta_only_ecg" "$SEED" --modality ecg --block-pattern delta
  run "engram_legacy_ecg"    "$SEED" --modality ecg --ssm-kind s4d_legacy --s4d-init lin

  # 1D CNN / Transformer baselines (same window + multi-label protocol).
  python scripts/train_baseline.py --model resnet1d --task ecg --epochs "$EPOCHS" --seed "$SEED" \
    --window-size 1000 --ecg-task superdiag --ecg-multilabel --data-root "$DATA_ROOT" \
    --output-dir "$RESULTS/resnet1d_ecg/seed$SEED" 2>&1 | tee "$RESULTS/resnet1d_ecg_seed${SEED}.log"
  python scripts/train_baseline.py --model transformer --task ecg --epochs "$EPOCHS" --seed "$SEED" \
    --window-size 1000 --ecg-task superdiag --ecg-multilabel --data-root "$DATA_ROOT" \
    --output-dir "$RESULTS/transformer_ecg/seed$SEED" 2>&1 | tee "$RESULTS/transformer_ecg_seed${SEED}.log"

  # Optional ablations (adds ~2× runtime). Enable with RUN_ABLATIONS=1.
  if [[ "$RUN_ABLATIONS" == "1" ]]; then
    run "abl_layers_6"    "$SEED" --modality ecg --num-layers 6
    run "abl_swa_on"      "$SEED" --modality ecg --layer-pattern s4,delta,swa,s4
  fi
done

echo "Done. Aggregate with: python scripts/aggregate_results.py $RESULTS --metric val_macro_auc"
