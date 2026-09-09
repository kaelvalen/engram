# ENGRAM: Modality-Portable Hybrid Sequence Architecture

A reference implementation evaluating a single hybrid SSD (Mamba-2) and Gated Delta Rule sequence backbone across 12-lead ECG (PTB-XL), speech (Speech Commands v2), and sequential images (CIFAR-10) without modality-specific structural modifications.

Nomenclature Note: This repository evaluates continuous linear-recurrent memory and dynamic token routing (SGMS). It is independent of DeepSeek's 2026 work *Engram: Conditional Memory via Scalable Lookup* (which uses static hashed N-gram lookup tables).

## 1. Architecture and Methodology

### Model Variations and Configurations

| Model ID | Layer Pattern | Mixer Family | Params (Laptop) | Params (Paper) | Description |
|---|---|---|---:|---:|---|
| `engram_hybrid` | SSD + Delta (3:1) | Mamba-2 SSD + Gated Delta | 258k | ~8M | Default hybrid backbone |
| `gateddelta_only` | Delta (All) | Gated Delta Rule | 185k | ~6M | Associative matrix memory backbone |
| `mamba2_only` | SSD (All) | Mamba-2 SSD | 282k | ~9M | Vector state selective scan |
| `engram_legacy` | S4D + Delta (3:1) | S4D-Lin + Gated Delta | 174k | ~6M | Diagonal S4D reference |
| `resnet1d` | 1D ResNet (4 blocks) | Standard Conv1D | 127k | - | Parameter efficiency reference |
| `resnet1d_wide` | 1D ResNet (92 ch) | Wide Conv1D | 255k | - | Parameter-matched CNN |
| `compact_cnn2d` | 2D ResNet (4 stages)| Standard Conv2D | 270k | - | Parameter-matched Vision |
| `audio_cnn` | 1D CNN (44 ch) | M5-derived Conv1D | 250k | - | Parameter-matched Audio |

### Mathematical Components and Ablation Matrix

| Component ID | Mechanism | Formulation | CLI Argument | Ablation Target |
|---|---|---|---|---|
| `local_conv` | Short causal conv | $x_c = \text{Conv1d}(x, k=4)$ | `--conv-kernel-size 0` | Local sequential context contribution |
| `out_gate` | Multiplicative gate | $o = o \odot \text{SiLU}(W_g x)$ | `--no-delta-out-gate` | Non-linear gating selectivity |
| `recurrent_mem`| Matrix state memory | $S_t = \alpha_t S_{t-1} + \beta_t v_t k_t^T$ | `--delta-memoryless` | Instantaneous attention vs memory ($S_{t-1}=0$) |
| `kv_memory` | Associative matrix | $o_t = S_t q_t$ | `--block-pattern delta` | Matrix ($d \times d$) vs vector ($d \times n$) SSM |
| `ffn_capacity` | SwiGLU expansion | $\text{SwiGLU}(x, e=2)$ | `--ffn-expand 0` | Token-level channel capacity |

## 2. Environment and Setup

### Dependencies and Installation
The system runs via Nix shell, CUDA 12/13, and uv:

```bash
# Clone repository and enter directory:
git clone https://github.com/kaelvalen/engram.git
cd engram

# Enter Nix environment (includes NVIDIA drivers and CUDA toolkit):
nix develop

# Alternative: Standard Python virtual environment (Python 3.12+):
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[train,test,gpu]"
```

* For NixOS GPU environments: `export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib`

## 3. Training and Inference (CLI)

All experiment outputs follow the pattern: `output/<run_id>/`

### ECG (PTB-XL, Multi-label Macro-AUROC)

```bash
# 1. Train hybrid model:
python train.py --modality ecg --ssm-kind ssd --ecg-multilabel --epochs 10 --batch-size 8 --hidden-dim 64 --num-layers 4 --seed 0 --output-dir output/engram_hybrid_ecg_seed0

# 2. Train parameter-matched ResNet1D-Wide:
python scripts/train_baseline.py --model resnet1d --base-channels 92 --task ecg --ecg-multilabel --epochs 10 --batch-size 16 --seed 0 --output-dir output/resnet1d_wide_ecg_seed0

# 3. Evaluate held-out test set:
python scripts/infer_ecg.py --checkpoint output/engram_hybrid_ecg_seed0/best_ecg.pt --ptbxl-test --output output/engram_hybrid_ecg_seed0/test_results.json
```

### Vision (Sequential CIFAR-10, Accuracy)

```bash
# 1. Train hybrid model:
python train.py --modality image --epochs 10 --batch-size 8 --hidden-dim 64 --num-layers 4 --patch-size 4 --seed 0 --output-dir output/engram_image_seed0

# 2. Train parameter-matched CompactConvNet2D:
python scripts/train_baseline.py --model compact_cnn2d --task image --epochs 10 --batch-size 64 --seed 0 --output-dir output/compact_cnn2d_seed0

# 3. Evaluate held-out test set:
python scripts/infer_image.py --checkpoint output/engram_image_seed0/best_image.pt --output output/engram_image_seed0/test_results.json
```

### Audio (Speech Commands v2, 10-Class Accuracy)

```bash
# 1. Prepare mel-spectrogram dumps:
python scripts/prepare_audio.py --source hf

# 2. Train hybrid model:
python train.py --modality audio --no-audio-synthetic --epochs 10 --batch-size 8 --hidden-dim 64 --num-layers 4 --seed 0 --output-dir output/engram_audio_seed0

# 3. Train parameter-matched AudioCNN:
python scripts/train_baseline.py --model audio_cnn --base-channels 44 --task audio --epochs 10 --batch-size 64 --seed 0 --output-dir output/audio_cnn_seed0
```

### Statistical Analysis and Benchmarking

```bash
# Aggregate multi-seed benchmark tables:
python scripts/aggregate_results.py output/benchmarks_laptop --metric val_macro_auc --detailed

# Compute paired t-tests, Hedges' g, and Evidence Tiers:
python scripts/statistical_analysis.py output/benchmarks_laptop --metric val_macro_auc --output output/benchmarks_laptop/stats_report.json
```

## 4. Metrics and Benchmarks

### Test Set Results (RTX 5060, Seeds 0-2 for ECG)

| Modality | Model ID | Parameters | Metric Type | Test Score (Mean ± Std) | Seed-level 95% CI | Evidence Tier (vs Reference) |
|---|---|---:|---|---:|---|---|
| **ECG** | `resnet1d` | 127k | Macro-AUROC | 0.9024 ± 0.0008 | [0.9016, 0.9032] | Reference Model |
| **ECG** | `resnet1d_wide` | 255k | Macro-AUROC | 0.9018 ± 0.0011 | [0.9006, 0.9030] | Parameter-Matched Reference |
| **ECG** | `gateddelta_only` | 185k | Macro-AUROC | 0.8978 ± 0.0048 | [0.8934, 0.9029] | Tier 3 (Inconclusive / No Detectable Difference, N=3) |
| **ECG** | `engram_hybrid` | 258k | Macro-AUROC | 0.8972 ± 0.0021 | [0.8956, 0.8996] | Tier 3 (Inconclusive / No Detectable Difference, N=3) |
| **ECG** | `engram_legacy` | 174k | Macro-AUROC | 0.8968 ± 0.0024 | [0.8941, 0.8982] | Tier 3 (Inconclusive / No Detectable Difference, N=3) |
| **ECG** | `mamba2_only` | 282k | Macro-AUROC | 0.8960 ± 0.0018 | [0.8943, 0.8979] | Tier 3 (Inconclusive / No Detectable Difference, N=3) |
| **ECG** | `transformer` | 3162k | Macro-AUROC | 0.8822 ± 0.0016 | [0.8810, 0.8840] | Tier 1 (Significantly Lower, p=0.0008) |
| **Image** | `engram_image` | 255k | Top-1 Accuracy | 72.38% (Seed 0) | - | Below CNN Baseline (~80% ResNet-18) |
| **Audio** | `engram_audio` | 258k | Top-1 Accuracy | 92.07% (Seed 0) | - | Matches CNN Baseline (~90% M5) |
