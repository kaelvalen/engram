# PRISM: A Modality-Portable Hybrid Linear-Recurrent Backbone for Clinical Time-Series and Natural Signals

**Target venue:** ICML 2026 ES-FoMo IV (4 pages main + unlimited refs/appendix,
ICML template) — backup: NeurIPS 2026 ENLSP-VI (4-page short). This file is the
prose skeleton; port to the ICML LaTeX template for submission. `[TODO]` marks
content that requires the experimental runs in `EXPERIMENTS.md`.

---

## Abstract

Hybrid linear-recurrent backbones (Mamba-2, Gated DeltaNet) dominate efficient
language modeling, but their design choices — tokenizers, head dims, layer
ratios — were tuned for text. We ask whether a *single* hybrid backbone, with no
modality-specific architectural changes and identical hyperparameters, can match
strong convolutional baselines across 12-lead ECG (PTB-XL), spoken commands, and
sequential images. PRISM interleaves Mamba-2-style SSD blocks (per-channel
selective state) with Gated Delta Rule blocks, optionally with sliding-window
attention. We contribute (i) a from-scratch, pure-PyTorch reference
implementation of the SSD scan and the chunked gated delta rule, with
numerical-equivalence tests against `torch.associative_scan` (verified on CPU)
and the FLA Triton kernels (gated by a GPU test — `[TODO: confirm pass before
claiming FLA equivalence]`); and (ii) a cross-modal portability study. `[TODO: headline result —
e.g. "PRISM matches xresnet1d101 within bootstrap CI on PTB-XL super-diagnostic
(0.9XX vs 0.928 macro AUC) while reusing the same backbone on audio and vision".]`

## 1. Introduction

The efficient-sequence-modeling literature has consolidated around hybrid
linear-recurrent architectures: Gated DeltaNet-H1/H2 (Yang et al., 2024) and the
Mamba family (Gu & Dao, 2023; Dao & Gu, 2024; Mamba-3, 2026) are deployed in
production LLMs. These designs are validated almost entirely on language. It is
unclear how much of their inductive bias transfers to continuous physiological
and natural signals, where convolutional models (e.g. `xresnet1d101` on PTB-XL)
remain strong, simple baselines.

We study a deliberately minimal question: **with one backbone and one set of
hyperparameters, how close can a modern SSD+Delta hybrid get to task-specific
CNNs across modalities?** Our contribution is engineering and empirical, not a
new architecture:

1. A clean from-scratch reference implementation of (a) the Mamba-2 SSD
   selective scan and (b) the chunked gated delta rule (UT-transform /
   triangular-solve), each numerically equivalent — verified by test — to the
   production `torch.associative_scan` (CPU-verified) and FLA kernels (GPU test
   `tests/test_delta_equivalence.py` is the gate — must pass before the FLA
   equivalence claim is made; the `g` mapping is per-step log-decay and is
   version-dependent). This is a reusable, auditable artifact for the
   linear-attention community.
2. A modality-portability study on PTB-XL (primary), Speech Commands, and
   sequential CIFAR-10, with the same backbone and no per-modality tuning, plus
   ablations isolating the per-channel-selectivity, layer-ratio, depth, and
   attention design choices.

## 2. Background

S4D (Gu et al., 2022) introduced diagonal complex SSMs; S6/Mamba added
input-dependent (selective) dynamics; Mamba-2 recast selection as state-space
duality (SSD) with scalar-per-head decay and per-channel state; Mamba-3 (ICLR
2026, poster + oral) added trapezoidal discretization, complex (RoPE-like) state
updates, and a MIMO formulation. In parallel, DeltaNet and Gated DeltaNet (Yang
et al., 2024) parallelize the delta rule over sequence length via the UT
transform, with fast FLA Triton kernels. Evidence that retrieval in SSM
hybrids depends on the attention layers (Michalak & Abreu, 2025) motivates our
optional sliding-window-attention ablation.

## 3. Method

**Backbone.** PRISM is a stack defined by a `block_pattern` of tokens
`{s4, delta, swa}` (default 3:1 SSD:Delta over 12 layers). Each block is a
pre-norm residual: `RMSNorm → (short causal conv) → mixer → +x`, then
`RMSNorm → SwiGLU → +x`. A single per-modality linear projection adapts input
dimensionality; a per-modality LayerNorm + linear head produces logits.

**SSD mixer.** For head `h` with per-channel input `x_t ∈ R^P`, scalar decay
`a_t = exp(Δ_t A_h)` (Δ_t per head, input-dependent; `A_h<0`), and per-head
selective vectors `B_t, C_t ∈ R^N`:
```
h_t = a_t h_{t-1} + (Δ_t x_t) ⊗ B_t ∈ R^{P×N},   y_t = ⟨h_t, C_t⟩ + D_h x_t.
```
Crucially the input is *not* averaged over `P` (the key difference from the
legacy S4D block, which collapsed Δ and u over the head dimension). The
recurrence is a first-order linear scan solved by `torch.associative_scan`.

**Gated delta rule.** Matrix state `S_t ∈ R^{P×P}` per head, with forget gate
`α_t` and write gate `β_t`:
`S_t = α_t[S_{t-1} − β_t (S_{t-1}k_t)k_t^⊤] + β_t v_t k_t^⊤`, `o_t = S_t q_t`.
Our reference implementation rewrites the intra-chunk recurrence as a lower-
triangular solve (UT transform), matching the published FLA kernel.

**Configurable layer pattern.** The ratio and placement of mixers is a single
config knob, enabling the ablations in §4 as a sweep rather than code changes.

## 4. Experiments

Setup: shared budget `hidden_dim=256, num_layers=12, num_heads=8` (~8M params),
AdamW + cosine, identical across modalities; mean ± std over 3 seeds. PTB-XL
metric is macro one-vs-rest AUROC (Strodthoff et al., 2020). `[TODO: run]`

**Main table.** `[TODO]` Architectures (ResNet1D, Transformer, SSD-only,
Delta-only, PRISM hybrid, PRISM legacy-S4D) × modalities (PTB-XL super-diag,
sCIFAR-10, Speech Commands).

**Ablations on PTB-XL super-diag.** `[TODO]`
- Layer pattern: 3:1, 1:1, 1:3, all-SSD, all-Delta, delta-top, delta-bottom.
- Depth: 6/12/18/24 layers.
- **Δ parameterisation: per-channel (SSD) vs per-head/mean-over-Dₕ (S4D).** Our
  central hypothesis is that per-channel selectivity is necessary.
- Sliding-window attention every 4 layers (H1-style): on/off.

**Throughput.** `[TODO: plot]` SSD scan: `associative_scan` vs reference vs
`torch.compile`; delta rule: reference vs FLA — across state dims N∈{16,64,128}.
Report on our own GPU (do not quote others' H100 numbers).

## 5. Discussion & Limitations

Be explicit and honest: (i) PRISM is *modality-portable* (same arch + HPs,
separate runs), not yet a single-set-of-weights joint model; (ii) `[TODO: state
which PTB-XL tasks we do/do not match within CI]`; (iii) the full 6-task
multi-label PTB-XL table and bootstrap CIs are `[TODO]`; (iv) we use Mamba-2 SSD
(stable) and cite Mamba-3 as future work.

## 6. Reproducibility

All code, the locked experiment matrix (`EXPERIMENTS.md`), seeds, and equivalence
tests are public. `pip install -e ".[test]"; pytest` runs 111 tests including
backend numerical-equivalence, float64 gradchecks, and streaming state-passing.

## References (to format in template)

- Gu, Goel, Ré. *Efficiently Modeling Long Sequences with Structured State Spaces (S4).* ICLR 2022.
- Gu, Gupta, Goel, Ré. *On the Parameterization and Initialization of Diagonal State Space Models (S4D).* NeurIPS 2022. arXiv:2206.11893.
- Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* 2023.
- Dao, Gu. *Transformers are SSMs: SSD / Mamba-2.* ICML 2024.
- Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu. *Mamba-3.* ICLR 2026 (poster + oral). arXiv:2603.15569.
- Yang, Kautz, Hatamizadeh. *Gated Delta Networks.* arXiv:2412.06464, ICLR 2025.
- Yang et al. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* NeurIPS 2024.
- Michalak, Abreu. *Some Attention is All You Need for Retrieval.* arXiv:2510.19861, NeurIPS 2025 workshop.
- Strodthoff et al. *Deep Learning for ECG Analysis: PTB-XL.* IEEE JBHI 2020. arXiv:2004.13701.
- Jaegle et al. *Perceiver IO.* ICLR 2022.
- Miralles-González et al. *You Can Train from Scratch.* ICLR 2025. arXiv:2501.14850.
