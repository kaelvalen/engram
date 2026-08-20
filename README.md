# ENGRAM — modality-portable hybrid linear-recurrent backbone

> One hybrid **SSD + Gated-Delta** sequence backbone — no modality-specific
> architecture — applied with **identical hyperparameters** to 12-lead ECG
> (PTB-XL), audio, and sequential images. A from-scratch reference
> implementation with full **numerical-equivalence tests** against the
> `torch.associative_scan` and FLA Triton kernels.
>
> Built on top of that verifiable core, **MoM** (Mixture of Memory Primitives)
> under `mom/` is the architecture contribution: a **surprise-gated router**
> decides, per token, which memory primitive updates its state — a learned
> answer to *when a compressed recurrent state suffices vs. when a surprising
> token must be written explicitly* (see [MoM](#mom--mixture-of-memory-primitives-mom)).
>
> *The Python package is still `engram` internally; "ENGRAM" is the project and
> paper name.*

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What's the claim

Hybrid linear-recurrent backbones (Mamba-2, Gated DeltaNet) win on *language*,
but their design choices are language-specific. ENGRAM tests whether a single
hybrid backbone — with **no modality-specific architectural tweaks** — matches
strong CNN baselines on PTB-XL (the primary, clinically-defensible target),
Speech Commands, and sequential CIFAR-10.

The backbone interleaves two complementary mixers (plus optional attention):

- **SSD blocks** (Mamba-2 state-space duality) — scalar-per-head decay with
  **per-channel state** and input-dependent (selective) Δ/B/C. This is the
  primitive Mamba-3 builds on, and the default `ssm_kind="ssd"`.
- **Gated Delta Rule blocks** — matrix-valued associative memory with
  data-dependent forget/write gates; targeted recall and overwrite.
- **Sliding-window attention** (optional, `swa`) — RoPE attention with a real
  **streaming KV-cache** (`SWAState`), so chunked/token-by-token decode is
  bit-exact with the full-sequence forward (fp64-tested). Used for H1-style
  hybrid ablations and as MoM's third expert.

The per-layer role tokens are defined in `engram.layer_tokens` as `("s4", "delta",
"swa")`. `engram.modules.block.BLOCK_REGISTRY` maps each token to a builder
function; new mixers can be registered with `@register_block("token")` without
changing the core config or model code.

> **Status (2026-07-18): architecture + training validated end-to-end on an
> RTX 5060 laptop GPU; the full paper matrix is not yet run.**
> 270+ tests pass (numerical equivalence, fp64 gradcheck, streaming
> state-passing, property-based). All six PTB-XL task vocabularies are
> validated against the real `scp_statements.csv` (5/23/44/19/12/71 classes —
> `scripts/validate_ptbxl_tasks.py`). The locked experiment matrix and honest
> gaps live in [EXPERIMENTS.md](EXPERIMENTS.md); paper skeleton at
> [paper/PAPER_DRAFT.md](paper/PAPER_DRAFT.md).

---

## Pipeline validation (laptop config, RTX 5060 — NOT paper numbers)

Protocol-identical but reduced (`hidden_dim=64, num_layers=4, batch 8`) run on
PTB-XL super-diag, **2 epochs, 1 seed**, macro-AUROC:

| Config | val macro-AUC |
|---|---|
| **ENGRAM hybrid (SSD+GDR)** | **0.8908** |
| Gated DeltaNet only | 0.8906 |
| ENGRAM legacy (S4D+GDR) | 0.8882 |
| Mamba-2 only (SSD) | 0.8836 |
| ResNet1D | 0.8828 |
| small Transformer | 0.8769 |

Reproduce with `DATA_ROOT=./datasets SEEDS="0" EPOCHS=2 bash scripts/run_benchmarks_laptop.sh`
then `python scripts/aggregate_results.py output/benchmarks_laptop --metric val_macro_auc`.
Do not quote these as the paper's main results — they are pipeline validation
at ~250 K params, 1 seed, 2 epochs.

## Install

### Option A: Nix + uv (recommended for GPU development)

```bash
git clone https://github.com/kaelvalen/engram.git
cd engram
nix develop
```

The dev shell provides `uv`, the CUDA toolkit, git, and just; Python deps are
installed into a local `.venv` by `uv` (declared in `pyproject.toml`). The
shell does **not** ship a Nix-built PyTorch — `uv` resolves PyTorch from PyPI
against your system NVIDIA driver.

### Option B: pip

```bash
git clone https://github.com/kaelvalen/engram.git
cd engram
pip install -e ".[train,test]"      # CPU-friendly: pure-PyTorch reference paths
pip install -e ".[gpu]"             # optional Triton kernels (FLA, mamba-ssm)
```

The pure-PyTorch reference paths run everywhere (no Triton/CUDA needed). The
`gpu` extra adds the production kernels; everything falls back gracefully if
they are absent or incompatible with the installed PyTorch/CUDA pair.

### NixOS GPU notes (PyPI wheels on NixOS)

PyPI torch wheels need host libraries that NixOS keeps out of default paths:

- `LD_LIBRARY_PATH` must include the NVIDIA driver libs (`/run/opengl-driver/lib`),
  a 64-bit `libstdc++.so.6`, and `libz.so.1`.
- Triton (for `torch.associative_scan` / FLA) calls `/sbin/ldconfig`, which does
  not exist on NixOS → set `TRITON_LIBCUDA_PATH=/run/opengl-driver/lib`.
- Triton's host compile needs `Python.h`, which the per-user profile does not
  expose → point `CPATH` at the store python's `include/python3.x` directory.

Verified working on NixOS + RTX 5060 (torch 2.13+cu130, FLA 0.3.2,
`tests/test_delta_equivalence.py` passes on GPU).

## Quickstart

```python
import torch
from engram import ENGRAMConfig, ModalityConfig, ENGRAMForClassification

cfg = ENGRAMConfig(
    hidden_dim=256,
    num_heads=8,
    num_layers=12,  # ~8M params, SSD+Delta 3:1
    ssm_kind="ssd",  # "ssd" (default) | "s4d_legacy"
    modalities=[
        ModalityConfig(name="ecg", input_dim=12, num_classes=5),
        ModalityConfig(name="image", input_dim=48, num_classes=10),
    ],
)
model = ENGRAMForClassification(cfg)

ecg = torch.randn(4, 1000, 12)  # 12-lead ECG
out = model(ecg, modality="ecg", labels=torch.randint(0, 5, (4,)))
print(out["loss"].item())
```

The layer mix is fully configurable:

```python
ENGRAMConfig(num_layers=12, block_pattern="s4,s4,s4,swa, s4,s4,s4,swa, s4,s4,s4,swa")  # H1-style
ENGRAMConfig(num_layers=4, force_block_type="delta")  # all-delta ablation
```

## Reproducing the paper

```bash
DATA_ROOT=./datasets SEEDS="0 1 2" EPOCHS=50 bash scripts/run_benchmarks.sh
python scripts/aggregate_results.py output/benchmarks      # mean ± std
python scripts/bench_throughput.py --device cuda --seq-len 4096
```

One command per table row; full matrix, datasets, metric (macro-AUROC) and
compute budget are in [EXPERIMENTS.md](EXPERIMENTS.md).

**Data prep:** PTB-XL goes under `datasets/ptbxl/` (validated by
`python scripts/validate_ptbxl_tasks.py`). Speech Commands mel dumps are built
by `python scripts/prepare_audio.py --source hf` (codec-free path that also
works behind TLS-intercepting proxies); output goes to `datasets/audio/{train,val}.pt`
(30,769 train + 7,777 val samples over the 10 core commands) and is consumed by
`engram/data/audio.get_audio_loaders(synthetic=False)`.

## MoM — Mixture of Memory Primitives (`mom/`)

MoM replaces ENGRAM's fixed block ratio with a **bank of heterogeneous memory
experts** (SSD, GDR; SWA scaffolded for v2) and a lightweight **per-token
router**: for every token, the router selects which memory primitive updates
its state and produces that token's output. The composition of the backbone
becomes a learned function of the token stream rather than a hand-tuned
constant (ENGRAM's 3:1 is recovered as the special case of a periodic router).

The router's learned projection (`z_t = W_r h_t`, Switch-style) is now
optionally augmented by an explicit **surprise signal** — how far each token
deviates from the recurrent state's prediction — so routing can ask *"is this
token surprising enough to write explicitly?"* before choosing a primitive.

- `mom/router.py` — TokenRouter (top-k, Switch-style; `learned` / `uniform` (B4) / `random` (B5) modes; optional surprise feature via `surprise_scale`)
- `mom/block.py` — MoMBlock: ENGRAM-exact residual/pre-norm anatomy + expert bank + router
- `mom/registry.py` — expert registry (`ssd`, `gdr`, `swa`), masked-execution contract
- `mom/losses.py` — Switch load-balancing + router z-loss (§3.7 stability objectives)
- `mom/baselines.py` — B1 fixed-3:1 hybrid, B2 SSD-only, B3 GDR-only, B4/B5 frozen routers
- `mom/tasks/` — MQAR (spike task), passkey, state-tracking probes
- `mom/analysis/` — §7 suite: routing heatmaps, specialization MI, knockout, composition, dynamics
- `configs/mom/` — spike / v1 / v1_k2 / v1_swa configs; `scripts/mom_spike_gate.py`, `scripts/mom_analysis.py`

Masked-dense execution (spec §3.4) is exact and fp64-tested: SSD decays on
every step (D1, `decay_on_skip`), GDR passes its state through exactly on a
miss, SWA's window slides over the routed subsequence. Sequential token-by-token
reference ≡ dense-masked forward at `rtol=1e-10`.

**Spike gate (§6.4), RTX 5060, 3 seeds × 8000 steps, MQAR 8-pairs @ T=64:**
routing does **not** collapse (min utilization 0.184 ≥ 0.10), but the quality
gate **fails** — MoM recall@4096 = 0.013 vs GDR-only 0.104 (SSD-only 0.000,
fixed-3:1 0.008). Read: uniform-ish early routing dilutes GDR on pure-recall
tasks. Ablations: `lambda_bal = 0` collapses hard (min util 0.0, R1 confirmed);
`shared_expert: ssd` keeps utilization healthy and one seed reaches 0.109
(B3-level) but not yet reliably. First C2 (specialization) signal is already
positive: all four layers show significant mutual information between expert
choice and token class (p = 0.002; layer 1: 0.52 nats). Run logs and
checkpoints under `output/mom/`.

### Surprise-gated routing (in progress)

The failure signature above (uniform-ish routing dilutes GDR on pure-recall)
points at a **weak routing signal** as much as a capacity limit: `z_t = W_r h_t`
carries no information about how well the recurrent state is predicting the
current token. The fix under test feeds a **normalized surprise estimator**
(EMA predictor + surprise scalar, in `engram/saber/`) into the router as an extra
per-token feature, so primitive choice can be driven by "is this token
surprising, i.e. worth an explicit write?"

- **Status — interface landed, experiment NOT run.** `TokenRouter` accepts an
  optional `[B, T]` `surprise` tensor gated by `surprise_scale` (default `0.0`
  ⇒ unchanged behaviour, all existing tests green), configurable via
  `MoMConfig.router_surprise_scale`.
- **Open question** (the next, separate task): where the surprise feature comes
  from at train time — a lightweight standalone predictor vs. the full SABER
  stack — and whether it closes the spike-gate gap to GDR-only. This is a
  *planned experiment*: no improvement is claimed before a run exists
  (see [EXPERIMENTS.md](EXPERIMENTS.md)).

`engram/saber/` supplies this signal: latent encoder → capacity-limited GRU
policy → EMA predictor → normalized surprise → adaptive budget → sparse slot
readout, with InfoNCE + JEPA-style losses, a 3-phase trainer, and diagnostics
(R1–R5). SABER is no longer a standalone "experimental" effort — it is the
surprise-signal provider inside MoM. Covered by `tests/test_saber.py`.

## Inference

```bash
# PTB-XL super-diagnostic (multi-label) or single-label task
python scripts/infer_ecg.py \
  --checkpoint output/ptbxl_superdiag/best.pt \
  --task superdiag \
  --output output/ptbxl_superdiag/test_report.json

# Sequential CIFAR-10
python scripts/infer_image.py \
  --checkpoint output/cifar10/best.pt \
  --output output/cifar10/test_report.json
```

## Reproducibility

- `--seed N` sets Python/NumPy/PyTorch RNGs; `--deterministic` enables
  `CUBLAS_WORKSPACE_CONFIG` for deterministic CUDA ops where possible.
- Checkpoints contain optimizer, scheduler, RNG state, and global step; resume
  with `--resume path/to/last.pt`.
- `last.pt` is written every epoch; `best.pt` is selected by the validation
  metric (macro-AUROC for ECG tasks, accuracy for image/audio).
- MoM runs are fully seeded (model init, router, data); identical reruns on the
  same hardware produce identical routing decisions and metrics.

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
| **ENGRAM (SSD + Delta hybrid)** | ~8M | _TODO_ | _TODO_ | _TODO_ |
| ENGRAM legacy (S4D + Delta) | ~8M | 0.884 acc *(prior, single-seed)* | _TODO_ | _TODO_ |

The legacy `0.884` sCIFAR number is from the previous S4D backbone, kept only
as a historical ablation row; see [EXPERIMENTS.md](EXPERIMENTS.md). All new
numbers must be **mean ± std over ≥3 seeds**. The 2-epoch laptop-validation
row above is the current state of the pipeline, not a submission number.

---

## Architecture

```
Input (any modality)  [B, T, input_dim]
        │  ModalityProjection  Linear(input_dim → hidden_dim)   ← per-modality
        ▼
ENGRAMBackbone   (block_pattern of s4 / delta / swa)
   s4    → SSDBlock     RMSNorm→Conv→SSD(selective scan)→res ; RMSNorm→SwiGLU→res
   delta → DeltaBlock   RMSNorm→Conv→GatedDeltaRule→res       ; RMSNorm→SwiGLU→res
   swa   → SWABlock     RMSNorm→SlidingWindowAttn(RoPE)→res   ; RMSNorm→SwiGLU→res
        │  mean / last pooling
        ▼  PerModalityHead  LayerNorm → Linear → logits
```

**SSD mixer** (`engram/modules/ssd.py`) — Mamba-2 state-space duality. `A` is a
scalar per head (`A=-exp(A_log)`), decay `a_t = exp(Δ_t·A)`, and crucially the
input is kept **per-channel** (`h_t = a_t h_{t-1} + (Δ_t x_t) ⊗ B_t`,
`y_t = ⟨h_t, C_t⟩ + D x_t`). No mean-over-Dₕ collapse — that was the central
weakness of the original S4D block, and the most important ablation in the
paper (`ssm_kind="ssd"` vs `"s4d_legacy"`).

**Parallel scan** (`engram/modules/scan.py`) — the recurrence is solved with
`torch.associative_scan` (fused HOP) or a vectorized two-buffer Hillis-Steele
fallback; neither uses strided indexed assignment. The original hand-derived
Blelloch up/down-sweep is preserved in `scan_reference.py` for teaching and as
an equivalence anchor.

**Gated delta rule** (`engram/modules/delta.py`) — `backend="reference"` is the
from-scratch chunked solve in the **division-free, log-space decay-ratio form**
(all coefficients `γ_t/γ_s ≤ 1`): the earlier `1/ᾱ` formulation underflowed to
inf and NaN'd in fp32 once the forget gate learned `α ≪ 1`. `backend="fla"`
calls FLA's `chunk_gated_delta_rule` Triton kernel with the correct
transposed-state mapping ((dk, dv) ↔ (dv, dk)) and falls back to the reference
if FLA/CUDA are unavailable.

**Sliding-window attention** (`engram/modules/attention.py`) — RoPE, causal
window, and a streaming KV-cache (`SWAState`: last `window` RoPE-applied
keys/values + absolute position). Chunked and token-by-token decode are
fp64-exact with full-sequence forward; the MoM masked path slides the window
over the routed subsequence only.

## Backends

| Component | `reference` (default) | production | fallback |
|---|---|---|---|
| SSD / S4D scan | Hillis-Steele (`scan_backend="reference"`) | `torch.associative_scan` (`"auto"`/`"assoc"`) + `torch.compile` | automatic to Hillis-Steele if `associative_scan` unavailable |
| Gated delta rule | pure-PyTorch chunked solve (division-free) | FLA `chunk_gated_delta_rule` (`delta_backend="fla"`) | automatic to reference if FLA missing, not on CUDA, wrong dtype, or Triton fails |

`tests/test_delta_equivalence.py` and `tests/test_scan_equivalence.py` assert
the production backends are numerically equivalent to the references. The FLA
equivalence test passes on the RTX 5060 with fla 0.3.2 (state-layout mapping
fixed 2026-07); it is skipped when the Triton kernel cannot execute.

## Testing

```bash
# inside nix develop, or with the pip venv activated
pytest                       # 270+ passed, 1 skipped (FLA probe skips when Triton unavailable)
ruff check engram mom tests scripts train.py
ruff format --check engram mom tests scripts train.py
```

- **Numerical equivalence** — scan/delta backends vs sequential reference; MoM dense-masked ≡ token-by-token reference at fp64 `rtol=1e-10`.
- **Gradcheck** (float64) — scan, SSD mixer, delta rule, full MoM block, masked SWA.
- **State-passing** — one-shot == chunked-with-carried-state for every mixer incl. SWA KV-cache (streaming correctness).
- **Regression** — seed-locked golden losses.
- **Property-based** (hypothesis) — finite outputs/loss/grads; gate simplex, mask idempotence, expert-order permutation invariance; CPU determinism.
- **FLA probe** — `tests/test_delta_equivalence.py` runs the FLA Triton kernel on a tiny tensor before declaring the backend available.
- **MoM suite** (`tests/mom/`) — routing, losses, masked equivalence, state passing, gradcheck, properties, tasks, training, analysis.

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
| `audio_synthetic` / `--audio-synthetic` | on | in-memory audio data; off = real Speech Commands dumps |

MoM flags live in `mom/config.py` (`experts`, `top_k`, `router_mode`,
`shared_expert`, `decay_on_skip`/`gdr_decay_on_skip`, `lambda_bal`, `lambda_z`,
`router_surprise_scale`).

---

## Repository layout

```
engram/
├── config.py                 # ENGRAMConfig (ssm_kind, block_pattern, backends, …)
├── layer_tokens.py           # dependency-free {s4, delta, swa} role tokens
├── model.py                  # projection → backbone → per-modality head
├── inference.py              # shared checkpoint loader for inference scripts
├── modules/
│   ├── ssd.py                # SSDMixer / SSDBlock  (Mamba-2 SSD, per-channel)
│   ├── s4.py                 # legacy S4D-Complex (ablation), parallel_scan wrapper
│   ├── delta.py              # GatedDeltaRule (division-free reference + FLA backends)
│   ├── attention.py          # SlidingWindowAttention / SWABlock (RoPE, KV-cache)
│   ├── scan.py               # associative_scan + two-buffer Hillis-Steele backends
│   ├── scan_reference.py     # preserved hand-derived Blelloch (teaching/equivalence)
│   └── block.py              # BLOCK_REGISTRY + ENGRAMBlock protocol + build_block dispatch
├── saber/                    # surprise-adaptive memory layer (supplies surprise → MoM routing)
├── training/                 # Trainer, CLI (train.py), metrics (macro-AUROC), loops
├── baselines/                # ResNet1D, small Transformer
└── data/                     # ecg / image / audio loaders (+ pure ptbxl_tasks mapping)
mom/                          # Mixture of Memory Primitives (see MoM section)
├── router.py masking.py block.py losses.py state.py registry.py config.py
├── model.py baselines.py reference.py train.py
├── tasks/                    # mqar.py, passkey.py, state_tracking.py
└── analysis/                 # heatmaps, specialization, knockout, composition, dynamics
configs/mom/                  # spike.yaml, v1.yaml, v1_k2.yaml, v1_swa.yaml
EXPERIMENTS.md                # locked benchmark matrix + honest gaps
paper/PAPER_DRAFT.md          # 4-page workshop manuscript skeleton
scripts/                      # run_benchmarks*.sh, bench_throughput.py, aggregate_results.py,
                              # infer_ecg.py, infer_image.py, validate_ptbxl_tasks.py,
                              # prepare_audio.py, mom_spike_gate.py, mom_analysis.py
tests/                        # pytest suite (+ tests/mom/, test_saber.py, test_swa_streaming.py)
```

## Honest scope

This is a **modality-portable architecture** (same arch + hyperparameters, one
training run per modality), not yet a single-set-of-weights joint model — that
true "modality-agnostic" result is the follow-up. The architecture is grounded
in the 2024–2026 frontier (Mamba-2/3, Gated DeltaNet, FLA); the from-scratch
reference implementations and their equivalence tests are the contribution
alongside the cross-modal portability study. MoM additionally treats the
*composition* of memory primitives as a learned, input-dependent decision
rather than a design constant. See [EXPERIMENTS.md](EXPERIMENTS.md) for what
still needs doing before submission.

## Citation

```bibtex
@misc{engram2026,
  title  = {ENGRAM: a modality-portable hybrid linear-recurrent backbone},
  author = {Hakbilen, Mehmet Arda},
  year   = {2026},
  note   = {https://github.com/kaelvalen/engram}
}
```

## Acknowledgments

Builds on ideas and (optionally) kernels from Mamba-2 / Mamba-3 (Dao, Gu et al.),
Gated DeltaNet (Yang, Kautz, Hatamizadeh et al.), and
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention).
