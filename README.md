# PRISM — modality-portable hybrid linear-recurrent backbone

> One hybrid **SSD + Gated-Delta** sequence backbone — no modality-specific
> architecture — applied with **identical hyperparameters** to 12-lead ECG
> (PTB-XL), audio, and sequential images. A from-scratch reference
> implementation with full **numerical-equivalence tests** against the
> `torch.associative_scan` and FLA Triton kernels.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What's the claim

Hybrid linear-recurrent backbones (Mamba-2, Gated DeltaNet) win on *language*,
but their design choices are language-specific. PRISM tests whether a single
hybrid backbone — with **no modality-specific architectural tweaks** — matches
strong CNN baselines on PTB-XL (the primary, clinically-defensible target),
Speech Commands, and sequential CIFAR-10.

The backbone interleaves two complementary mixers (plus optional attention):

- **SSD blocks** (Mamba-2 state-space duality) — scalar-per-head decay with
  **per-channel state** and input-dependent (selective) Δ/B/C. This is the
  primitive Mamba-3 builds on, and the default `ssm_kind="ssd"`.
- **Gated Delta Rule blocks** — matrix-valued associative memory with
  data-dependent forget/write gates; targeted recall and overwrite.
- **Sliding-window attention** (optional, `swa`) — for H1-style hybrid
  ablations, since "some attention" is what carries retrieval in SSM hybrids.

> **Heads-up — this is a research pivot in progress.** The architecture and
> tooling are implemented and tested; the *benchmark numbers are not filled in
> yet* (they need GPU + datasets). See [EXPERIMENTS.md](EXPERIMENTS.md) for the
> locked matrix and honest gaps, and [paper/PAPER_DRAFT.md](paper/PAPER_DRAFT.md)
> for the manuscript skeleton.

---

## Install

```bash
git clone https://github.com/kaelvalen/prism.git
cd prism
pip install -e ".[train,test]"      # CPU-friendly: pure-PyTorch reference paths
pip install -e ".[gpu]"             # optional Triton kernels (FLA, mamba-ssm)
```

The pure-PyTorch reference paths run everywhere (no Triton/CUDA needed). The
`gpu` extra adds the production kernels; everything falls back gracefully if
they are absent.

## Quickstart

```python
import torch
from prism import PRISMConfig, ModalityConfig, PRISMForClassification

cfg = PRISMConfig(
    hidden_dim=256, num_heads=8, num_layers=12,   # ~8M params, SSD+Delta 3:1
    ssm_kind="ssd",                               # "ssd" (default) | "s4d_legacy"
    modalities=[
        ModalityConfig(name="ecg",   input_dim=12, num_classes=5),
        ModalityConfig(name="image", input_dim=48, num_classes=10),
    ],
)
model = PRISMForClassification(cfg)

ecg = torch.randn(4, 1000, 12)                    # 12-lead ECG
out = model(ecg, modality="ecg", labels=torch.randint(0, 5, (4,)))
print(out["loss"].item())
```

The layer mix is fully configurable:

```python
PRISMConfig(num_layers=12, block_pattern="s4,s4,s4,swa, s4,s4,s4,swa, s4,s4,s4,swa")  # H1-style
PRISMConfig(num_layers=4,  force_block_type="delta")                                  # all-delta ablation
```

## Reproducing the paper

```bash
DATA_ROOT=./datasets SEEDS="0 1 2" EPOCHS=50 bash scripts/run_benchmarks.sh
python scripts/aggregate_results.py output/benchmarks      # mean ± std
python scripts/bench_throughput.py --device cuda --seq-len 4096
```

One command per table row; full matrix, datasets, metric (macro-AUROC) and
compute budget are in [EXPERIMENTS.md](EXPERIMENTS.md).

## Benchmark numbers

Primary metric on PTB-XL is **macro one-vs-rest AUROC** (Strodthoff et al. 2020),
not accuracy. Baseline to match: `xresnet1d101` ≈ **0.928** macro AUC on the
5-class super-diagnostic task (within ±0.005 bootstrap CI).

| Model | params | sCIFAR-10 acc | PTB-XL super-diag AUC | Speech Cmds acc |
|---|---|---|---|---|
| ResNet1D (xresnet1d101) | ~8M | – | 0.928 *(lit.)* | – |
| Small Transformer | ~8M | _TODO_ | _TODO_ | _TODO_ |
| Mamba-2 only (SSD) | ~8M | _TODO_ | _TODO_ | _TODO_ |
| Gated DeltaNet only | ~8M | _TODO_ | _TODO_ | _TODO_ |
| **PRISM (SSD + Delta hybrid)** | ~8M | _TODO_ | _TODO_ | _TODO_ |
| PRISM legacy (S4D + Delta) | ~8M | 0.884 acc *(prior, single-seed)* | _TODO_ | _TODO_ |

The legacy `0.884` sCIFAR number is from the previous S4D backbone, kept only
as a historical ablation row; see [EXPERIMENTS.md](EXPERIMENTS.md). All new
numbers must be **mean ± std over ≥3 seeds**.

---

## Architecture

```
Input (any modality)  [B, T, input_dim]
        │  ModalityProjection  Linear(input_dim → hidden_dim)   ← per-modality
        ▼
PRISMBackbone   (block_pattern of s4 / delta / swa)
   s4    → SSDBlock     RMSNorm→Conv→SSD(selective scan)→res ; RMSNorm→SwiGLU→res
   delta → DeltaBlock   RMSNorm→Conv→GatedDeltaRule→res       ; RMSNorm→SwiGLU→res
   swa   → SWABlock     RMSNorm→SlidingWindowAttn(RoPE)→res   ; RMSNorm→SwiGLU→res
        │  mean / last pooling
        ▼  PerModalityHead  LayerNorm → Linear → logits
```

**SSD mixer** (`prism/modules/ssd.py`) — Mamba-2 state-space duality. `A` is a
scalar per head (`A=-exp(A_log)`), decay `a_t = exp(Δ_t·A)`, and crucially the
input is kept **per-channel** (`h_t = a_t h_{t-1} + (Δ_t x_t) ⊗ B_t`,
`y_t = ⟨h_t, C_t⟩ + D x_t`). No mean-over-Dₕ collapse — that was the central
weakness of the original S4D block, and the most important ablation in the
paper (`ssm_kind="ssd"` vs `"s4d_legacy"`).

**Parallel scan** (`prism/modules/scan.py`) — the recurrence is solved with
`torch.associative_scan` (fused HOP) or a vectorized Hillis-Steele fallback;
neither uses strided indexed assignment. The original hand-derived Blelloch
up/down-sweep is preserved in `scan_reference.py` for teaching and as an
equivalence anchor.

**Gated delta rule** (`prism/modules/delta.py`) — `backend="reference"` is the
from-scratch chunked solve (UT transform via `solve_triangular`);
`backend="fla"` calls FLA's `chunk_gated_delta_rule` Triton kernel and falls
back to the reference if FLA/CUDA are unavailable.

## Backends

| Component | `reference` (default) | production |
|---|---|---|
| SSD / S4D scan | Hillis-Steele (`scan_backend="reference"`) | `torch.associative_scan` (`"auto"`/`"assoc"`) + `torch.compile` |
| Gated delta rule | pure-PyTorch chunked solve | FLA `chunk_gated_delta_rule` (`delta_backend="fla"`) |

`tests/test_delta_equivalence.py` and `tests/test_scan_equivalence.py` assert
the production backends are numerically equivalent to the references (the FLA
case runs on GPU; it is skipped on CPU CI).

## Testing

```bash
pip install -e ".[test]"
pytest                       # 111 tests: equivalence, shapes, gradcheck, state-passing, regression, property-based
ruff check prism tests scripts train.py
```

- **Numerical equivalence** — scan backends and delta backends vs sequential/reference ground truth.
- **Gradcheck** (float64) — scan, SSD mixer, delta rule.
- **State-passing** — one-shot == chunked-with-carried-state (streaming correctness).
- **Regression** — seed-locked golden losses.
- **Property-based** (hypothesis) — finite outputs/loss/grads across random shapes; CPU determinism.

## Key config flags

| Field / CLI flag | Default | Description |
|---|---|---|
| `ssm_kind` / `--ssm-kind` | `ssd` | `ssd` (Mamba-2 selective) or `s4d_legacy` |
| `block_pattern` / `--layer-pattern` | `None` | explicit per-layer tokens `s4,delta,swa` (overrides interleave) |
| `delta_every` | 4 | DeltaBlock every Nth layer (3:1) when no explicit pattern |
| `delta_backend` / `--delta-backend` | `reference` | `reference` or `fla` |
| `scan_backend` / `--scan-backend` | `auto` | `auto` / `assoc` / `reference` |
| `s4d_init` / `--s4d-init` | `lin` | S4D-Lin (`A=-½+iπn`) or `legacy` |
| `swa_window` / `--swa-window` | 128 | sliding-window attention span |
| `compile` / `--compile` | off | `torch.compile` the model in the trainer |
| `multilabel` / `--ecg-multilabel` | off | PTB-XL multi-label targets + BCE + macro-AUROC selection |

---

## Repository layout

```
prism/
├── config.py                 # PRISMConfig (ssm_kind, block_pattern, backends, …)
├── model.py                  # projection → backbone → per-modality head
├── modules/
│   ├── ssd.py                # SSDMixer / SSDBlock  (Mamba-2 SSD, per-channel)
│   ├── s4.py                 # legacy S4D-Complex (ablation), parallel_scan wrapper
│   ├── delta.py              # GatedDeltaRule (reference + FLA backends)
│   ├── attention.py          # SlidingWindowAttention / SWABlock (RoPE)
│   ├── scan.py               # associative_scan + Hillis-Steele scan backends
│   ├── scan_reference.py     # preserved hand-derived Blelloch (teaching/equivalence)
│   └── block.py              # build_block / forward_block dispatch
├── training/                 # Trainer, CLI (train.py), metrics (macro-AUROC), loops
├── baselines/                # ResNet1D, small Transformer
└── data/                     # ecg / image / audio loaders
EXPERIMENTS.md                # locked benchmark matrix + honest gaps
paper/PAPER_DRAFT.md          # 4-page workshop manuscript skeleton
scripts/                      # run_benchmarks.sh, bench_throughput.py, aggregate_results.py
tests/                        # pytest suite
```

## Honest scope

This is a **modality-portable architecture** (same arch + hyperparameters, one
training run per modality), not yet a single-set-of-weights joint model — that
true "modality-agnostic" result is the follow-up. The architecture is grounded
in the 2024–2026 frontier (Mamba-2/3, Gated DeltaNet, FLA); the from-scratch
reference implementations and their equivalence tests are the contribution
alongside the cross-modal portability study. See [EXPERIMENTS.md](EXPERIMENTS.md)
for what still needs doing before submission.

## Citation

```bibtex
@misc{prism2026,
  title  = {PRISM: a modality-portable hybrid linear-recurrent backbone},
  author = {Hakbilen, Mehmet Arda},
  year   = {2026},
  note   = {https://github.com/kaelvalen/prism}
}
```

## Acknowledgments

Builds on ideas and (optionally) kernels from Mamba-2 / Mamba-3 (Dao, Gu et al.),
Gated DeltaNet (Yang, Kautz, Hatamizadeh et al.), and
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention).
