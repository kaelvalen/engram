# Cross-Modal Portability of Linear-Recurrent Hybrids and Dynamic Heterogeneous Memory Routing

Target venue: ICML 2026 ES-FoMo IV (4 pages main + unlimited refs/appendix).

## Abstract

Linear-recurrent sequence models (Mamba-2 SSD and Gated DeltaNet) achieve competitive language modeling efficiency, but their ability to serve as a general-purpose, modality-agnostic backbone for non-language signals remains an open empirical question. We evaluate a fixed hybrid backbone interleaving State-Space Duality (SSD) and Gated Delta Rule (GDR) associative memory across 12-lead ECG (PTB-XL), spoken commands (Speech Commands v2), and sequential images (CIFAR-10) without modality-specific structural modifications. While the fixed hybrid demonstrates cross-modal execution, task-specialized convolutional baselines (ResNet1D-Wide) remain competitive or superior on 1D physiological waveforms at matched parameter counts (~255k). To move beyond static composition, we investigate State-Gated Memory Selection (SGMS): token-level conditional allocation across heterogeneous memory dynamics (vector SSM, associative matrix, and local window attention). We identify a critical mathematical pathology in top-1 renormalised routing where task loss gradients are eliminated ($\partial \mathcal{L}_{\text{task}} / \partial W_r = 0$), and show that a straight-through estimator restores task-directed optimization. We present an 11-arm ablation framework evaluating memory update semantics, surprise-conditioned routing, and systems efficiency tradeoffs.

## 1. Introduction

Recurrent and state-space architectures have gained traction as sub-quadratic alternatives to Transformers. Mamba-2 (Dao and Gu, ICML 2024) introduced State-Space Duality (SSD), unifying continuous state-space models with structured masked attention. Concurrently, Gated DeltaNet (Yang et al., ICLR 2025) augmented linear attention with a hardware-efficient chunked delta rule for associative recall.

Existing literature frequently claims architectural novelty by combining linear-recurrent layers in fixed interleaving patterns. However, static hybrids (such as fixed 3:1 SSD:GDR ratios) do not constitute a fundamentally new architecture. Their defensible scientific contribution lies in evaluating cross-modal portability: whether a single sequence backbone, tuned without domain-specific spatial or spectral convolutions, transfers effectively to continuous physiological, audio, and visual modalities.

### Nomenclature and Disambiguation
We distinguish this work from DeepSeek's concurrent architecture *Engram: Conditional Memory via Scalable Lookup* (arXiv:2601.07372, 2026). DeepSeek's model utilizes static hashed N-gram lookup tables to retrieve frozen lexical representations. In contrast, our system operates on dynamic recurrent hidden states through continuous state-space decay and online delta-rule associative updates.

### Beyond Static Composition: Heterogeneous Memory Routing
Static alternation raises an unresolved question: why should token sequence processing adhere to a rigid periodic schedule? Recent mixture-of-experts architectures explore token routing in recurrent models:
- Routing Mamba (2025) conditionally routes across homogeneous SSM parameterizations.
- Mixture-of-Memories (ICLR 2026) routes tokens across independent recurrent state vectors.
- MambaFormer (2026) and FlowHN (ACL 2026) route tokens between full attention and SSM layers.

We propose State-Gated Memory Selection (SGMS). SGMS routes tokens not among duplicated parameters or between attention and generic SSMs, but across heterogeneous sequence mixers with fundamentally different memory update dynamics:
1. Distributed vector state-space (SSD): continuous exponential decay tracking long-range global state.
2. Associative matrix state (GDR): online key-value error correction via the delta rule.
3. Sliding window attention (SWA): local episodic buffer over short contexts.

## 2. Fixed Hybrid Baseline Substrate

### Mathematical Formulation
The baseline sequence backbone processes inputs through stacked blocks. Each block applies RMSNorm, short causal 1D convolution ($k=4$), a linear-recurrent mixer, and a SwiGLU feedforward network with residual connections.

For SSD (State-Space Duality), head $h$ with input $x_t \in \mathbb{R}^P$, scalar decay $a_t = \exp(\Delta_t A_h)$, and selective vectors $B_t, C_t \in \mathbb{R}^N$ evolves as:
$$h_t = a_t h_{t-1} + (\Delta_t x_t) \otimes B_t \in \mathbb{R}^{P \times N}$$
$$y_t = \langle h_t, C_t \rangle + D_h x_t$$

For the Gated Delta Rule (GDR), associative matrix state $S_t \in \mathbb{R}^{P \times P}$ evolves under forget gate $\alpha_t \in (0, 1)$ and write rate $\beta_t \in (0, 1)$:
$$S_t = \alpha_t [S_{t-1} - \beta_t (S_{t-1} k_t) k_t^T] + \beta_t v_t k_t^T$$
$$o_t = S_t q_t$$
where intra-chunk recurrence is solved as a lower-triangular system via the unit-triangular (UT) transform.

### Empirical Portability on Continuous Modalities
We evaluate the fixed hybrid across three modalities using identical core sequence parameters (`hidden_dim=64, num_layers=4, num_heads=4`, ~258k parameters for exploratory scale; target full scale is 256x12, ~8M parameters):
1. **12-lead ECG (PTB-XL)**: 1000 timesteps (10s at 100 Hz). The hybrid achieves 0.8972 macro-AUROC. However, the parameter-matched convolutional baseline (`resnet1d_wide`, 255k parameters) achieves 0.9018 macro-AUROC. Classical 1D CNN inductive bias remains competitive on structured physiological waveforms.
2. **Sequential Vision (CIFAR-10)**: Non-overlapping 4x4 patches (64 tokens). The hybrid reaches 72.38% top-1 accuracy. This demonstrates sequence modeling transfer, but does not surpass 2D spatial convolution networks (~80%+).
3. **Spoken Commands (Speech Commands v2)**: Mel-spectrogram sequence modeling reaches 92.07% top-1 accuracy, matching standard audio CNN baselines (M5-derived, 90.5%).

These results establish that the fixed hybrid serves as a viable cross-modal sequence modeling substrate, but does not provide universal dominance over domain-tuned convolutional baselines.

## 3. SGMS: Learned Heterogeneous Memory Routing

### Router Formulation and the Zero-Gradient Trap
Let router logits for token $t$ across $K$ memory primitives be:
$$z_t = W_r h_t + \lambda_s w_e \cdot s_t$$
where $W_r \in \mathbb{R}^{K \times D}$, $s_t$ is an optional scalar token surprise signal, and $w_e \in \mathbb{R}^K$ are per-expert coefficients. Softmax probabilities are $p_t = \text{softmax}(z_t)$.

For top-$k$ selection, let $S_t = \text{topk}(z_t, k)$ and selected probabilities be $\tilde{p}_t = \text{gather}(p_t, S_t)$. Renormalised routing gates are:
$$g_t = \frac{\tilde{p}_t}{\sum_{j \in S_t} \tilde{p}_{t, j}}$$

**The Top-1 Gradient Elimination Defect**: When $k=1$, $S_t = \{i^*\}$, and the renormalised gate reduces to:
$$g_t = \frac{p_{t, i^*}}{p_{t, i^*}} \equiv 1.0$$
Consequently:
$$\frac{\partial g_t}{\partial z_t} = 0$$
The task loss $\mathcal{L}_{\text{task}}$ exerts zero gradient on the router weights:
$$\frac{\partial \mathcal{L}_{\text{task}}}{\partial W_r} = \frac{\partial \mathcal{L}_{\text{task}}}{\partial y_t} \cdot \text{expert}_{i^*}(h_t) \cdot \frac{\partial g_t}{\partial W_r} = 0$$
Under this condition, router weights $W_r$ are driven entirely by auxiliary regularization:
$$\mathcal{L}_{\text{router}} = \lambda_{\text{bal}} \mathcal{L}_{\text{bal}} + \lambda_z \mathcal{L}_z$$
The router optimizes load balancing and numerical stability, but fails to learn task-directed memory primitive allocation.

### Straight-Through Gate Estimator
To enable task-directed gradient flow for discrete $k=1$ routing, we incorporate a Straight-Through Estimator (STE):
$$g_t^{\text{STE}} = \text{stop\_gradient}(g_t) + \tilde{p}_t - \text{stop\_gradient}(\tilde{p}_t)$$
In the forward pass, $g_t^{\text{STE}} = 1.0$, preserving discrete top-1 execution. In the backward pass, $\partial g_t^{\text{STE}} / \partial \tilde{p}_t = 1.0$, allowing task loss gradients $\partial \mathcal{L}_{\text{task}} / \partial y_t$ to propagate directly through the softmax into $W_r$.

### Memory Update Semantics on Skipped Tokens
In standard MoE networks, inactive experts are bypassed. In recurrent memory, bypassing introduces a temporal semantic choice when token $t$ misses expert $e$:
1. **Wall-Clock Continuous Decay (`decay_on_skip=True`)**: The recurrent state decays as if an empty token passed ($h_t = a_t h_{t-1}$). The expert tracks global sequence elapsed time.
2. **Expert-Local Event Time (`decay_on_skip=False`)**: The recurrent state freezes ($h_t = h_{t-1}$). The expert tracks compressed, expert-local sub-sequences.

For GDR associative matrix memory, exact state freezing ($S_t = S_{t-1}$) prevents spurious associative forgetting during skip intervals.

### Surprise-Conditioned Routing
We test whether token information novelty informs memory demand. Given a one-step linear predictor $\hat{h}_t = W_p h_{t-1}$, token surprise is measured as normalized prediction error:
$$s_t = \frac{\|h_t - \hat{h}_t\|_2}{\sqrt{D}}$$
Because adding a uniform scalar to all logits is softmax and argmax shift-invariant ($z + c \mathbf{1} \equiv z$), surprise must be modulated by learned per-expert parameters $w_e$:
$$z_{t, e} = W_{r, e} h_t + \lambda_s w_e s_t$$
Under this formulation, tokens with low surprise ($s_t \approx 0$) route preferentially to vector decay state (SSD), while unpredictable tokens ($s_t \gg 0$) route to high-capacity matrix associative memory (GDR).

## 4. Empirical Evaluation Framework

### Benchmark Protocol and Statistical Rigor
For PTB-XL ECG, evaluation utilizes macro one-vs-rest AUROC on 1000-timestep full waveforms across 3 seeds. Pairwise differences are tested via paired t-tests and Hedges' g effect sizes with Holm-Bonferroni corrections.

Under an $N=3$ protocol, failure to reject the null hypothesis ($p \ge 0.05$) or small empirical effect sizes ($|g| < 0.2$) reflects low inferential power ($1 - \beta \approx 0.10$). Such results cannot support claims of equivalence. Establishing practical equivalence requires larger seed counts ($N \ge 10$) and Two One-Sided Tests (TOST) against an a priori practical equivalence margin.

| Architecture | Parameters | Val Macro-AUROC | Test Macro-AUROC | Evidence Classification (vs ResNet1D) |
|---|---:|---:|---:|---|
| `resnet1d` | 127k | 0.9024 ± 0.0008 | - | Reference Model |
| `resnet1d_wide` | 255k | 0.9018 ± 0.0011 | - | Parameter-Matched Reference |
| `gateddelta_only` | 185k | 0.8978 ± 0.0048 | 0.8908 ± 0.0013 | Inconclusive / No Detectable Difference (p=0.27) |
| `engram_hybrid` | 258k | 0.8972 ± 0.0021 | 0.8929 ± 0.0022 | Inconclusive / Directional Trend (p=0.07) |
| `engram_legacy` | 174k | 0.8968 ± 0.0024 | 0.8911 ± 0.0013 | Inconclusive / Directional Trend (p=0.09) |
| `mamba2_only` | 282k | 0.8960 ± 0.0018 | 0.8907 ± 0.0048 | Statistically Lower (p=0.02) |
| `transformer` | 3162k | 0.8822 ± 0.0016 | - | Statistically Lower (p=0.0008) |

### SGMS 11-Arm Ablation Matrix
To isolate routing specialization from generic capacity, SGMS is structured into an 11-arm comparative matrix on the Multi-Query Associative Recall (MQAR) benchmark:
- `B1`: Fixed 3:1 SSD:GDR hybrid baseline.
- `B2`: SSD-only homogeneous baseline.
- `B3`: GDR-only homogeneous baseline.
- `B4`: SGMS learned router with straight-through gradient estimation ($k=1$).
- `B5`: SGMS uniform routing (fixed $1/K$ gate distribution).
- `B6`: SGMS random per-token routing.
- `B7`: SGMS learned routing + token surprise feature $s_t$.
- `B8`: SGMS + shuffled surprise (temporal permutation test).
- `B9`: SGMS + inverted surprise ($-s_t$ directionality test).
- `B10`: Hard top-1 selection.
- `B11`: Dense top-2 mixture across $K=2$ experts.

Contrasting `B4` vs `B5` evaluates whether dynamic token allocation outperforms static uniform distribution. `B4` vs `B6` determines whether learned assignments capture task structure rather than stochastic regularization. `B7` vs `B8` confirms whether surprise provides semantic signal rather than auxiliary scalar capacity.

### Systems and Computational Tradeoffs
In SGMS v1, all $K$ experts execute over full sequence length $T$, with outputs masked by routing gates ($K \times T$ total compute). This confirms mathematical correctness and dynamic path selection, but does not reduce FLOPs. Realizing computational efficiency requires SGMS v2 gathered execution: tokens assigned to expert $e$ are gathered into contiguous buffers ($T_e \le T$), processed through specialized Triton kernels, and scattered back to sequence order.

## 5. Conclusion

ENGRAM evaluates the cross-modal sequence modeling capability of hybrid linear-recurrent primitives across continuous physical and natural signals. While the fixed 3:1 SSD:GDR hybrid executes stably across modalities, task-specialized convolutional architectures remain competitive on 1D biological signals. SGMS addresses the rigidity of fixed composition by framing sequence modeling as learned routing across heterogeneous memory dynamics. By resolving the top-1 gradient elimination defect via straight-through estimation and formalizing memory update semantics on skipped tokens, SGMS establishes a principled foundation for dynamic, heterogeneous memory allocation in recurrent neural networks.
