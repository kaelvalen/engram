#!/usr/bin/env bash
# PRISM benchmark matrix tuned for NVIDIA RTX 5090 (Blackwell / sm_120).
#
# Assumes a single GPU with >=24 GB VRAM. The RTX 5090 runs the full paper
# config (hidden_dim=256, num_layers=12, num_heads=8, ~9.9M params) at batch
# size 32 across all three modalities. Batch size 64 OOMs on 32 GB because the
# SSD scan tensor scales linearly with batch size.
#
# Blackwell-specific notes:
# - PyTorch >=2.12 with cuDNN 9.9+ / CUDA 13.0+ is required for sm_120 support.
# - torch.compile is enabled by default; disable with COMPILE=0 if you hit
#   compilation issues on the very first run.
# - flash-linear-attention's Triton kernels may not yet support sm_120, so this
#   script stays on the reference delta backend and torch.associative_scan SSD
#   backend (the default), which are numerically equivalent.
#
# Usage:
#   DATA_ROOT=./datasets SEEDS="0 1 2" EPOCHS=50 bash scripts/run_benchmarks_rtx5090.sh
set -euo pipefail

# Blackwell / large-VRAM friendly memory allocator.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${DATA_ROOT:-./datasets}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-50}"
AMP="${AMP:-bf16}"
BATCH_SIZE="${BATCH_SIZE:-32}"
COMPILE="${COMPILE:-1}"
RESULTS="${RESULTS:-output/benchmarks_rtx5090}"
mkdir -p "$RESULTS"

# Full paper backbone budget (~8M params).
COMMON="--hidden-dim 256 --num-layers 12 --num-heads 8 --batch-size $BATCH_SIZE --amp $AMP --data-root $DATA_ROOT --epochs $EPOCHS --gradient-checkpointing"
[[ "$COMPILE" == "1" ]] && COMMON="$COMMON --compile"

run () {  # run <name> <seed> <extra-args...>
  local name="$1"; local seed="$2"; shift 2
  # PTB-XL is evaluated multi-label (macro AUROC) per the paper protocol.
  local extra=""
  case " $* " in *" --modality ecg "*) extra="--ecg-multilabel";; esac
  echo ">>> [$name | seed=$seed] $* $extra"
  PYTHONHASHSEED="$seed" prism-train $COMMON --seed "$seed" --output-dir "$RESULTS/$name/seed$seed" "$@" $extra \
    2>&1 | tee "$RESULTS/${name}_seed${seed}.log"
}

for SEED in $SEEDS; do
  # ---------------- Main table: architectures × modalities ----------------
  for MOD in ecg image audio; do
    run "prism_hybrid_$MOD"    "$SEED" --modality "$MOD" --ssm-kind ssd
    run "mamba2_only_$MOD"     "$SEED" --modality "$MOD" --ssm-kind ssd --block-pattern s4
    run "gateddelta_only_$MOD" "$SEED" --modality "$MOD" --block-pattern delta
    run "prism_legacy_$MOD"    "$SEED" --modality "$MOD" --ssm-kind s4d_legacy --s4d-init lin
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
