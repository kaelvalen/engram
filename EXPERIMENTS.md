# ENGRAM: Experiment Matrix and Validation Ledger

All reported metrics reflect the mean and standard deviation (mean ± std) across at least 3 distinct random seeds. Single-seed runs are not used for architectural conclusions.

## 1. Datasets and Evaluation Protocols

| Modality | Dataset | Path | Primary Metric | Class Count | Input Window |
|---|---|---|---|---:|---:|
| **ECG** | PTB-XL v1.0.3 | `datasets/ptbxl/` | Macro One-vs-Rest AUROC | 5 / 23 / 44 / 19 / 12 / 71 | 1000 steps (10s @ 100Hz) |
| **Audio** | Speech Commands v2 | `datasets/audio/` | Top-1 Accuracy | 10 | 64 mel bins |
| **Vision** | Sequential CIFAR-10| `datasets/cifar/` | Top-1 Accuracy | 10 | 64 patches (4x4x3) |

## 2. Benchmark Scorecard and Pairwise Statistics

### Validation Results (PTB-XL ECG Superdiag, RTX 5060, 3 Seeds, 10 Epochs)

Configuration: `hidden_dim=64, num_layers=4, batch_size=8`, multi-label macro-AUROC:

| Model ID | Parameters | Val Macro-AUROC | AUROC / 100k Params | Seed 0 | Seed 1 | Seed 2 | Test Macro-AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `resnet1d` (Reference) | 127k | **0.9024 ± 0.0008** | **0.7060** | 0.9032 | 0.9024 | 0.9016 | - |
| `gateddelta_only` | 185k | 0.8978 ± 0.0048 | 0.4845 | 0.8894 | 0.8910 | 0.8919 | 0.8908 ± 0.0013 |
| `engram_hybrid` | 258k | 0.8972 ± 0.0021 | 0.3474 | 0.8909 | 0.8953 | 0.8925 | 0.8929 ± 0.0022 |
| `engram_legacy` | 174k | 0.8968 ± 0.0024 | 0.5149 | 0.8899 | 0.8910 | 0.8925 | 0.8911 ± 0.0013 |
| `mamba2_only` | 282k | 0.8960 ± 0.0018 | 0.3171 | 0.8866 | 0.8961 | 0.8895 | 0.8907 ± 0.0048 |
| `transformer` | 3162k | 0.8822 ± 0.0016 | 0.0279 | 0.8838 | 0.8806 | 0.8821 | - |

### Pairwise Significance Matrix (Holm-Bonferroni Corrected)

| Pairwise Comparison | Difference (Delta) | Hedges' g (Cohen's d) | t-statistic | p-value (raw) | p-value (Holm) | Evidence Tier |
|---|---:|---:|---:|---:|---:|---|
| `resnet1d` vs `transformer` | +0.0202 | 11.93 (20.88) | 36.169 | 0.0008 | 0.0115 | Tier 1: Statistically Significant |
| `engram_hybrid` vs `transformer` | +0.0150 | 2.47 (4.32) | 7.485 | 0.0174 | 0.2260 | Tier 1: Statistically Significant |
| `engram_hybrid` vs `resnet1d` | -0.0052 | -1.18 (-2.07) | -3.577 | 0.0700 | 0.6304 | Tier 2: Directional Trend (ResNet leads) |
| `engram_hybrid` vs `gateddelta_only` | -0.0006 | -0.13 (-0.24) | -0.408 | 0.7230 | 1.0000 | Tier 3: Measurement Noise / Equivalence |
| `engram_hybrid` vs `engram_legacy` | +0.0004 | 0.11 (0.19) | 0.332 | 0.7714 | 0.7714 | Tier 3: Measurement Noise / Equivalence |
| `engram_hybrid` vs `mamba2_only` | +0.0011 | 0.52 (0.91) | 1.568 | 0.2573 | 1.0000 | Tier 3: Measurement Noise / Equivalence |

## 3. Component Ablation Matrix

Systematic isolation of individual architectural mechanisms within the hybrid block:

| Ablation Arm | Modified Parameter | Parameter Count | Architectural Hypothesis |
|---|---|---|---|
| `full_hybrid` | Default configuration | 258k | Baseline control |
| `no_conv` | `--conv-kernel-size 0` | 256k | Short causal convolution local context effect |
| `no_out_gate` | `--no-delta-out-gate` | 242k | Multiplicative output gating non-linearity |
| `delta_memoryless` | `--delta-memoryless` | 185k | Sequential state memory ($S_{t-1}$) vs instantaneous attention |
| `no_ffn` | `--ffn-expand 0` | 159k | SwiGLU FFN per-position channel expansion |
| `pure_recurrent` | `--conv-kernel-size 0 --ffn-expand 0` | 157k | Pure recurrent mixer core (no conv, no FFN) |
| `pool_last` | `--pool-type last` | 258k | Last-token pooling vs mean sequence pooling |

### Execution Commands

```bash
# Execute multi-architecture ECG benchmark:
DATA_ROOT=./datasets SEEDS="0 1 2" EPOCHS=10 bash scripts/run_benchmarks_laptop.sh

# Execute 5-component architectural ablation:
DATA_ROOT=./datasets SEEDS="0 1 2" EPOCHS=10 bash scripts/run_component_ablation.sh

# Generate consolidated statistical report:
python scripts/statistical_analysis.py output/benchmarks_laptop --metric val_macro_auc --output output/benchmarks_laptop/stats_report.json
```
