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

| Model ID | Parameters | Val Macro-AUROC | Seed 0 | Seed 1 | Seed 2 | Test Macro-AUROC |
|---|---:|---:|---:|---:|---:|---:|
| `resnet1d` (Reference) | 127k | **0.9024 ± 0.0008** | 0.9032 | 0.9024 | 0.9016 | - |
| `resnet1d_wide` (Matched) | 255k | 0.9018 ± 0.0011 | 0.9029 | 0.9012 | 0.9013 | - |
| `gateddelta_only` | 185k | 0.8978 ± 0.0048 | 0.8894 | 0.8910 | 0.8919 | 0.8908 ± 0.0013 |
| `engram_hybrid` | 258k | 0.8972 ± 0.0021 | 0.8909 | 0.8953 | 0.8925 | 0.8929 ± 0.0022 |
| `engram_legacy` | 174k | 0.8968 ± 0.0024 | 0.8899 | 0.8910 | 0.8925 | 0.8911 ± 0.0013 |
| `mamba2_only` | 282k | 0.8960 ± 0.0018 | 0.8866 | 0.8961 | 0.8895 | 0.8907 ± 0.0048 |
| `transformer` | 3162k | 0.8822 ± 0.0016 | 0.8838 | 0.8806 | 0.8821 | - |

Note: Bounded non-linear metrics such as AUROC cannot be meaningfully divided by parameter counts. Model efficiency is evaluated through parameter-matched baselines (ResNet1D-Wide at 255k matching ENGRAM at 258k) and hardware execution latency.

### Pairwise Significance Matrix (Holm-Bonferroni Corrected)

Under N=3 seeds, failure to reject the null hypothesis (p >= 0.05) or small effect sizes (|g| < 0.2) indicates inconclusive statistical power, not mathematical or practical equivalence. Formal equivalence requires Two One-Sided Tests (TOST) against an a priori equivalence margin.

| Pairwise Comparison | Difference (Delta) | Hedges' g (Cohen's d) | t-statistic | p-value (raw) | p-value (Holm) | Evidence Tier |
|---|---:|---:|---:|---:|---:|---|
| `resnet1d` vs `transformer` | +0.0202 | 11.93 (20.88) | 36.169 | 0.0008 | 0.0115 | Tier 1: Statistically Significant |
| `engram_hybrid` vs `transformer` | +0.0150 | 2.47 (4.32) | 7.485 | 0.0174 | 0.2260 | Tier 1: Statistically Significant |
| `engram_hybrid` vs `resnet1d` | -0.0052 | -1.18 (-2.07) | -3.577 | 0.0700 | 0.6304 | Tier 2: Directional Trend (ResNet leads) |
| `engram_hybrid` vs `gateddelta_only` | -0.0006 | -0.13 (-0.24) | -0.408 | 0.7230 | 1.0000 | Tier 3: Inconclusive / No Detectable Difference (N=3) |
| `engram_hybrid` vs `engram_legacy` | +0.0004 | 0.11 (0.19) | 0.332 | 0.7714 | 0.7714 | Tier 3: Inconclusive / No Detectable Difference (N=3) |
| `engram_hybrid` vs `mamba2_only` | +0.0011 | 0.52 (0.91) | 1.568 | 0.2573 | 1.0000 | Tier 3: Inconclusive / No Detectable Difference (N=3) |

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

## 4. SGMS: Dynamic Heterogeneous Memory Routing Matrix

SGMS evaluates dynamic token-level allocation across distinct memory update dynamics: vector SSM state (SSD), associative matrix memory (GDR), and local sliding window attention (SWA).

### Experimental Ablation Matrix (B1-B11)

| Arm | Description | Routing Configuration | Target Hypothesis |
|---|---|---|---|
| `B1` | Fixed 3:1 Hybrid | Static interleaving (SSD:GDR) | Baseline fixed substrate |
| `B2` | SSD Only | All layers SSD (vector SSM) | Homogeneous vector memory baseline |
| `B3` | GDR Only | All layers GDR (associative matrix) | Homogeneous matrix associative baseline |
| `B4` | SGMS Learned | Learned router, top_k=1, straight_through=True | Task-adaptive dynamic primitive selection |
| `B5` | SGMS Uniform | Fixed equal probability (1/K) | Disentangles router learning from dynamic execution |
| `B6` | SGMS Random | Random uniform per-token routing | Tests whether learned allocation outperforms stochastic routing |
| `B7` | SGMS + Surprise | Learned router + surprise feature $s_t$ | Tests information novelty as a routing signal |
| `B8` | SGMS + Shuffled Surprise | Surprise sequence randomly permuted | Tests temporal semantic validity vs scalar parameter capacity |
| `B9` | SGMS + Inverted Surprise | Negative surprise feature ($-s_t$) | Verifies directional hypothesis (high novelty -> associative memory) |
| `B10`| SGMS Top-1 | Hard selection (top_k=1) | Sparse dynamic allocation |
| `B11`| SGMS Top-2 | Soft mixture across K=2 experts | Dense mixture baseline vs sparse selection |

### Hypothesis Test Contrasts

1. **Learned Routing Value (`B4` vs `B5`)**: Measures whether data-driven token routing improves over uniform distribution.
2. **Non-Random Specialization (`B4` vs `B6`)**: Verifies that learned assignment correlates with token task structure rather than stochastic regularization.
3. **Surprise Signal Utility (`B7` vs `B4`)**: Evaluates whether prediction error from local context informs memory primitive demand.
4. **Directionality of Surprise (`B7` vs `B9`)**: Confirms whether surprising tokens preferentially route to high-capacity associative memory.
5. **Semantic Relevance of Novelty (`B7` vs `B8`)**: Rules out the confound that surprise merely introduces an uncalibrated scalar parameter.

### Systems and Computational Execution

1. **SGMS v1 (Dense Masked Execution)**: All K experts execute over the full sequence length T. Outputs are masked according to routing decisions. This establishes mathematical viability and dynamic path selection, but does not provide FLOP reduction.
2. **SGMS v2 (Gathered Execution Roadmap)**: Tokens are dynamically gathered by expert assignment ($T_{\text{selected}}$ per expert), dispatched to individual memory kernels, and scattered back to sequence order. This delivers true conditional compute efficiency.
