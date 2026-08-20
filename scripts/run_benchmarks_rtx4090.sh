#!/usr/bin/env bash
# ENGRAM benchmark matrix tuned for NVIDIA RTX 4090 (Ada Lovelace / sm_89, 24 GB).
#
# The 4090 runs the FULL paper config (hidden_dim=256, num_layers=12,
# num_heads=8, ~8M params) at batch 24. Unlike Blackwell (see the 5090 script),
# the 4090 has mature full support for PyTorch, torch.compile, and Triton:
#   - torch.associative_scan (SSD) and FLA Triton delta kernels both work, so
#     production backends are usable (reference remains the safe default and is
#     numerically equivalent; set DELTA_BACKEND=fla to opt into Triton).
#   - No sm_120/cuDNN workarounds needed.
#
# Memory: the SSD scan tensor scales linearly with batch size, so 24 GB is the
# comfortable default; raise to 32 only if it fits in your VRAM.
#
# Usage:
#   DATA_ROOT=./datasets SEEDS="0 1 2" EPOCHS=50 bash scripts/run_benchmarks_rtx4090.sh
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${DATA_ROOT:-./datasets}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-50}"
AMP="${AMP:-bf16}"
BATCH_SIZE="${BATCH_SIZE:-24}"
COMPILE="${COMPILE:-1}"
DELTA_BACKEND="${DELTA_BACKEND:-reference}"  # reference (safe) | fla (Triton)
RESULTS="${RESULTS:-output/benchmarks_rtx4090}"
mkdir -p "$RESULTS"

# Full paper backbone budget (~8M params), SSD via associative_scan (default).
COMMON="--hidden-dim 256 --num-layers 12 --num-heads 8 --batch-size $BATCH_SIZE --amp $AMP --data-root $DATA_ROOT --epochs $EPOCHS --gradient-checkpointing --delta-backend $DELTA_BACKEND"
[[ "$COMPILE" == "1" ]] && COMMON="$COMMON --compile"

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
  # ---------------- Main table: architectures × modalities ----------------
  for MOD in ecg image audio; do
    run "engram_hybrid_$MOD"    "$SEED" --modality "$MOD" --ssm-kind ssd
    run "mamba2_only_$MOD"     "$SEED" --modality "$MOD" --ssm-kind ssd --block-pattern s4
    run "gateddelta_only_$MOD" "$SEED" --modality "$MOD" --block-pattern delta
    run "engram_legacy_$MOD"    "$SEED" --modality "$MOD" --ssm-kind s4d_legacy --s4d-init lin
  done

  # CNN / Transformer baselines (same window + multi-label protocol).
  python scripts/train_baseline.py --model resnet1d --task ecg --epochs "$EPOCHS" --seed "$SEED" \
    --batch-size "$BATCH_SIZE" --window-size 1000 --ecg-task superdiag --ecg-multilabel \
    --data-root "$DATA_ROOT" --output-dir "$RESULTS/resnet1d_ecg/seed$SEED" \
    2>&1 | tee "$RESULTS/resnet1d_ecg_seed${SEED}.log"
  python scripts/train_baseline.py --model transformer --task ecg --epochs "$EPOCHS" --seed "$SEED" \
    --batch-size "$BATCH_SIZE" --window-size 1000 --ecg-task superdiag --ecg-multilabel \
    --data-root "$DATA_ROOT" --output-dir "$RESULTS/transformer_ecg/seed$SEED" \
    2>&1 | tee "$RESULTS/transformer_ecg_seed${SEED}.log"

  # ---------------- Ablations (on PTB-XL super-diag) ----------------
  run "abl_pattern_3to1"   "$SEED" --modality ecg --layer-pattern s4,s4,s4,delta,s4,s4,s4,delta,s4,s4,s4,delta
  run "abl_pattern_1to1"   "$SEED" --modality ecg --layer-pattern s4,delta,s4,delta,s4,delta,s4,delta,s4,delta,s4,delta
  run "abl_pattern_1to3"   "$SEED" --modality ecg --layer-pattern s4,delta,delta,delta,s4,delta,delta,delta,s4,delta,delta,delta
  run "abl_all_ssd"        "$SEED" --modality ecg --block-pattern s4
  run "abl_all_delta"      "$SEED" --modality ecg --block-pattern delta
  run "abl_delta_top"      "$SEED" --modality ecg --layer-pattern s4,s4,s4,s4,s4,s4,s4,s4,s4,delta,delta,delta
  run "abl_delta_bottom"   "$SEED" --modality ecg --layer-pattern delta,delta,delta,s4,s4,s4,s4,s4,s4,s4,s4,s4

  for L in 6 12 18 24; do run "abl_layers_$L" "$SEED" --modality ecg --num-layers "$L"; done

  run "abl_delta_perchannel" "$SEED" --modality ecg --ssm-kind ssd
  run "abl_delta_perhead"    "$SEED" --modality ecg --ssm-kind s4d_legacy

  run "abl_swa_on"  "$SEED" --modality ecg --layer-pattern s4,s4,s4,swa,s4,s4,s4,swa,s4,s4,s4,swa
  run "abl_swa_off" "$SEED" --modality ecg --ssm-kind ssd

done

echo "Done. Aggregate with: python scripts/aggregate_results.py $RESULTS --metric val_macro_auc"
