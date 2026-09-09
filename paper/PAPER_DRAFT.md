# ENGRAM: Modality-Portable Hybrid Linear-Recurrent Backbone

Target venue: ICML 2026 ES-FoMo IV (4 pages main + unlimited refs/appendix).

## Abstract

Hybrid linear-recurrent backbones (Mamba-2, Gated DeltaNet) were developed and tuned for natural language processing. This work evaluates whether a single hybrid architecture, without modality-specific structural modifications, matches convolutional baselines on 12-lead ECG (PTB-XL), spoken commands (Speech Commands v2), and sequential images (CIFAR-10). ENGRAM interleaves Mamba-2 state-space duality (SSD) blocks with Gated Delta Rule associative memory blocks. Contributions include: (1) an independent PyTorch reference implementation of SSD and chunked gated delta rule with numerical equivalence tests against CUDA/Triton kernels; (2) a multi-modal portability benchmark with paired statistical tests, Holm-Bonferroni corrections, and parameter-efficiency metrics. On PTB-XL ECG, ENGRAM variants perform within measurement noise of each other (mean AUROC 0.896-0.898, p > 0.25, Tier 3 equivalence), while a task-specific 1D ResNet baseline achieves 0.9024 AUROC with 2x higher parameter efficiency.

## 1. Introduction

Efficient sequence modeling literature has centered on hybrid linear-recurrent architectures. Gated DeltaNet-H1/H2 and the Mamba family demonstrate state-of-the-art efficiency in language modeling. Transfer of these architectural priors to continuous physiological and natural signals remains unverified, where convolutional models remain standard baselines.

This paper tests a specific question: With a fixed backbone and constant hyperparameters, how close does an SSD+Delta hybrid get to task-specific convolutional networks across modalities?

Contributions:
1. Reference implementation of Mamba-2 SSD selective scan and chunked Gated Delta Rule, verified against production Triton kernels.
2. Modality-portability evaluation across PTB-XL ECG, Speech Commands v2, and CIFAR-10 with identical core hyperparameters.
3. Component isolation ablation covering local convolution, output gating, recurrent state memory, associative memory, and feedforward expansion.

## 2. Method

### Backbone Architecture
ENGRAM defines layer patterns via block tokens `(s4, delta, swa)` with a default 3:1 SSD:Delta ratio over 12 layers. Each block uses pre-normalization: `RMSNorm -> ShortCausalConv1d -> Mixer -> Residual`, followed by `RMSNorm -> SwiGLU -> Residual`. Modality adapters project raw inputs to the hidden dimension, and pooling heads produce task logits.

### SSD Selective Scan
For head $h$ with per-channel input $x_t \in \mathbb{R}^P$, scalar decay $a_t = \exp(\Delta_t A_h)$, and selective vectors $B_t, C_t \in \mathbb{R}^N$:
$$h_t = a_t h_{t-1} + (\Delta_t x_t) \otimes B_t \in \mathbb{R}^{P \times N}$$
$$y_t = \langle h_t, C_t \rangle + D_h x_t$$
Inputs retain channel dimension $P$ without averaging, solved via parallel associative scan.

### Gated Delta Rule
Matrix associative state $S_t \in \mathbb{R}^{P \times P}$ with forget gate $\alpha_t$ and write gate $\beta_t$:
$$S_t = \alpha_t [S_{t-1} - \beta_t (S_{t-1} k_t) k_t^T] + \beta_t v_t k_t^T$$
$$o_t = S_t q_t$$
The intra-chunk recurrence is solved as a lower-triangular system via the UT transform.

## 3. Empirical Results and Discussion

### Experimental Protocol
Shared configuration: `hidden_dim=64, num_layers=4, num_heads=4` for pipeline validation, `hidden_dim=256, num_layers=12, num_heads=8` for full paper scale. Evaluation uses macro one-vs-rest AUROC on PTB-XL (full 10-second signals at 100 Hz, 1000 timesteps) and Top-1 accuracy on audio and vision.

### PTB-XL Results (3 Seeds, Mean ± Std)

| Architecture | Parameters | Val Macro-AUROC | AUROC / 100k Param | Evidence Tier (vs ResNet1D) |
|---|---:|---:|---:|---|
| `resnet1d` | 127k | 0.9024 ± 0.0008 | 0.7060 | Reference Model |
| `gateddelta_only` | 185k | 0.8978 ± 0.0048 | 0.4845 | Tier 3 (Equivalence, p=0.27) |
| `engram_hybrid` | 258k | 0.8972 ± 0.0021 | 0.3474 | Tier 2 (Directional Trend, p=0.07) |
| `engram_legacy` | 174k | 0.8968 ± 0.0024 | 0.5149 | Tier 2 (Directional Trend, p=0.09) |
| `mamba2_only` | 282k | 0.8960 ± 0.0018 | 0.3171 | Tier 1 (Lower, p=0.02) |
| `transformer` | 3162k | 0.8822 ± 0.0016 | 0.0279 | Tier 1 (Significantly Lower, p=0.0008) |

### Findings
1. Equivalence among recurrent variants: Differences between `engram_hybrid`, `gateddelta_only`, and `engram_legacy` are within measurement noise ($|g| \le 0.13, p > 0.7$).
2. Parameter efficiency: `resnet1d` achieves higher AUROC with 51% fewer parameters than `engram_hybrid`.
3. Multi-modal portability: The identical hybrid backbone executes across ECG (0.893 test AUROC), audio (92.07% accuracy), and vision (72.38% accuracy) without architectural modification.
