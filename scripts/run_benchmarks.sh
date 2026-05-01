#!/bin/bash
set -e

echo "========================================="
echo "    PRISM Ablation & Benchmark Suite"
echo "========================================="

RESULTS_DIR="output/benchmarks"
mkdir -p $RESULTS_DIR

# -------------------------------
# 1) CIFAR-10 Ablations
# -------------------------------
echo ">>> Running CIFAR-10 Ablation: Hybrid (S4 + Delta)"
python scripts/train_image.py \
    --config configs/train.example.yaml \
    --epochs 5 \
    --output-dir "$RESULTS_DIR/hybrid_cifar10" | tee $RESULTS_DIR/cifar10_hybrid.log

echo ">>> Running CIFAR-10 Ablation: All-S4"
python scripts/train_image.py \
    --config configs/train.example.yaml \
    --epochs 5 \
    --block-pattern s4 \
    --output-dir "$RESULTS_DIR/alls4_cifar10" | tee $RESULTS_DIR/cifar10_alls4.log

echo ">>> Running CIFAR-10 Ablation: All-Delta"
python scripts/train_image.py \
    --config configs/train.example.yaml \
    --epochs 5 \
    --block-pattern delta \
    --output-dir "$RESULTS_DIR/alldelta_cifar10" | tee $RESULTS_DIR/cifar10_alldelta.log

# -------------------------------
# 2) ECG comparisons
# -------------------------------
echo ">>> Running ECG Benchmark: PRISM (Hybrid)"
python scripts/train_ecg.py \
    --config configs/train.example.yaml \
    --epochs 5 \
    --output-dir "$RESULTS_DIR/prism_ecg" | tee $RESULTS_DIR/ecg_prism.log

echo ">>> Running ECG Benchmark: ResNet1D Baseline"
python scripts/train_baseline.py \
    --model resnet1d \
    --task ecg \
    --epochs 5 \
    --output-dir "$RESULTS_DIR/resnet1d_ecg" | tee $RESULTS_DIR/ecg_resnet1d.log

echo ">>> Running ECG Benchmark: Transformer Baseline"
python scripts/train_baseline.py \
    --model transformer \
    --task ecg \
    --epochs 5 \
    --output-dir "$RESULTS_DIR/transformer_ecg" | tee $RESULTS_DIR/ecg_transformer.log

echo "========================================="
echo " Benchmarks completed. Results are saved in $RESULTS_DIR"
echo "========================================="
