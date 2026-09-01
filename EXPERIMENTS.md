# ENGRAM — Experiment Matrix

This file *locks* the experiment plan before running anything (per good ML
practice). Every table row in the paper maps to one command here. Report
**mean ± std over ≥3 seeds**, never single-best.

## Datasets

Place under `$DATA_ROOT` (default `./datasets`):

| Modality | Dataset | Where | Loader |
|---|---|---|---|
| ECG (primary) | PTB-XL 1.0.3 | physionet.org/content/ptb-xl/ → `datasets/ptbxl/` | `engram/data/ecg.py` |
| Audio (secondary) | Speech Commands v2 | torchaudio download → `datasets/audio/` | `engram/data/audio.py` |
| Vision (tertiary) | sequential CIFAR-10 | torchvision auto-download → `datasets/cifar/` | `engram/data/image.py` |

## Primary metric

PTB-XL is benchmarked with **macro one-vs-rest AUROC**, not accuracy
(Strodthoff et al. 2020). Use `engram.training.metrics.roc_auc_ovr_macro` /
`engram.training.loops.evaluate_macro_auc`. Target: match `xresnet1d101`
(~0.928 macro AUC on 5-class super-diagnostic) within the ±0.005 bootstrap CI.

**Use the full signal.** `--window-size` defaults to **1000** = the whole 10 s
record at 100 Hz, matching what `xresnet1d101` consumes. Shorter windows (the old
250/128 defaults) truncate to the first 1–2.5 s, discard most of the ECG, and
systematically bias ENGRAM *down* relative to the baseline — an apples-to-oranges
confound. Keep 1000 for any number quoted against the leaderboard (use 5000 at
500 Hz). This costs ~4× the sequence length vs the old default, but ENGRAM is a
linear-time model so it is O(T); correctness over speed here.

## Main table (architectures × modalities)

Shared budget: `hidden_dim=256, num_layers=12, num_heads=8` (~8M params).

| Model | Command (per seed) |
|---|---|
| ENGRAM hybrid (SSD + Delta 3:1) | `train.py --modality <m> --ssm-kind ssd` |
| Mamba-2 only (SSD) | `train.py --modality <m> --ssm-kind ssd --block-pattern s4` |
| Gated DeltaNet only | `train.py --modality <m> --block-pattern delta` |
| ENGRAM legacy (S4D + Delta) | `train.py --modality <m> --ssm-kind s4d_legacy --s4d-init lin` |
| ResNet1D baseline | `scripts/train_baseline.py --model resnet1d --task ecg --window-size 1000 --ecg-task superdiag --ecg-multilabel` |
| Transformer baseline | `scripts/train_baseline.py --model transformer --task ecg --window-size 1000 --ecg-task superdiag --ecg-multilabel` |

`<m> ∈ {ecg, image, audio}`. **Same hyperparameters, no per-modality tuning** —
that portability is the claim.

## Ablations (on PTB-XL super-diag — cheapest)

1. **Layer pattern** (12 layers): 3:1, 1:1, 1:3, all-SSD, all-Delta, delta-top,
   delta-bottom — via `--layer-pattern`.
2. **Depth**: `--num-layers {6,12,18,24}`.
3. **Δ parameterisation** (the most important ablation): per-channel selective
   (`--ssm-kind ssd`) vs per-head / mean-over-Dₕ (`--ssm-kind s4d_legacy`). This
   directly tests the review's central critique of the original S4D block.
4. **Sliding-window attention every 4 layers** (H1-style hybrid): on/off via
   `--layer-pattern s4,s4,s4,swa,...`.

## Throughput plot

```
python scripts/bench_throughput.py --device cuda --seq-len 4096
```
Compares `torch.associative_scan` vs reference vs `torch.compile` for the SSD
scan, and reference vs FLA for the delta rule, across state dims N∈{16,64,128}.
Report **your hardware's** numbers — do not quote the paper's H100 figures.

## Running it all

```
DATA_ROOT=./datasets SEEDS="0 1 2" EPOCHS=50 bash scripts/run_benchmarks.sh
python scripts/aggregate_results.py output/benchmarks --metric val_macro_auc   # mean ± std
```

Budget ≈ 30–60 single-GPU-hours (4090/A100/5090). If constrained: drop sCIFAR,
run 2 seeds for ablations.

## RTX 4090 / Ada

For an NVIDIA RTX 4090 (24 GB, Ada/sm_89) use the dedicated script. Full paper
config (~8M params) at default batch 24, `torch.compile` on, expandable-segments
allocator. Unlike Blackwell, the 4090 has full Triton support: the SSD
`associative_scan` backend is used and the FLA delta backend is available
(`DELTA_BACKEND=fla`; default `reference` is the safe, equivalence-tested
choice):

```
DATA_ROOT=./datasets SEEDS="0 1 2" EPOCHS=50 bash scripts/run_benchmarks_rtx4090.sh
python scripts/aggregate_results.py output/benchmarks_rtx4090 --metric val_macro_auc
```

## RTX 5090 / Blackwell

For an NVIDIA RTX 5090 use the dedicated script. It runs the full paper matrix
with `torch.compile` enabled and the expandable-segments allocator. Default
batch size is 32 (batch size 64 OOMs on 32 GB because the SSD scan scales
linearly with batch size):

```
DATA_ROOT=./datasets SEEDS="0 1 2" EPOCHS=50 bash scripts/run_benchmarks_rtx5090.sh
python scripts/aggregate_results.py output/benchmarks_rtx5090 --metric val_macro_auc
```

Override batch size with `BATCH_SIZE=16` if needed. Requirements for sm_120
(Blackwell):
- PyTorch >=2.12 built against CUDA 13.0+ / cuDNN 9.9+.
- `flash-linear-attention`'s Triton kernels may not yet support sm_120, so the
  script keeps the default reference delta backend and `torch.associative_scan`
  SSD backend. Disable compile with `COMPILE=0` if the first run fails during
  `torch.compile` warm-up.

## Running on a constrained GPU (≤8 GB laptop)

The paper matrix (`hidden_dim=256, num_layers=12, num_heads=8`) needs ~16 GB
VRAM. On an 8 GB laptop GPU the default `run_benchmarks.sh` will OOM during the
first training step. Use the reduced script instead:

```
DATA_ROOT=./datasets SEEDS="0" EPOCHS=10 bash scripts/run_benchmarks_laptop.sh
python scripts/aggregate_results.py output/benchmarks_laptop --metric val_macro_auc
```

This keeps the **same protocol** (full 10 s ECG, multi-label macro-AUROC) but
shrinks the backbone to `hidden_dim=64, num_layers=4, num_heads=4` and uses
batch size 8. It is meant for **pipeline validation and quick iteration only**;
do not quote these numbers as the paper's main results. Add optional ablations
with `RUN_ABLATIONS=1` (adds ~2× runtime).

## Honest gaps / TODO before submission

- **All six PTB-XL task groups ARE wired** (`--ecg-task superdiag|subdiag|diag|
  form|rhythm|all`): multi-hot targets in `engram/data/ecg.py` via the pure,
  unit-tested `engram/data/ptbxl_tasks.py` mapping, BCEWithLogits in `model.py`,
  accumulating macro AUROC in `loops.evaluate_multilabel_auc`, checkpoint
  selection on `val_macro_auc`, and `num_classes` derived from the task vocab.
  Each non-superdiag task is inherently multi-label. The mapping logic is tested
  on a synthetic SCP table (`tests/test_ptbxl_tasks.py`) **and validated against
  the real `scp_statements.csv`** by `scripts/validate_ptbxl_tasks.py`:
  vocab sizes are exactly 5 / 23 / 44 / 19 / 12 / 71 for superdiag/subdiag/
  diag/form/rhythm/all, and all 21,799 records' SCP codes resolve.
- **Bootstrap CIs** are implemented (`metrics.bootstrap_auroc_ci`, 1000-resample,
  matches Strodthoff Table I `0.928(05)` format). Run it on the **test** fold for
  the final number; the in-training AUROC uses the val fold.
- **FLA / mamba-ssm numbers** require a GPU; `tests/test_delta_equivalence.py`
  must pass there before trusting `--delta-backend fla`. The `g` argument we pass
  is the **per-step** log-decay (`g_t = log α_t`); the FLA op forms the cumulative
  decays internally. This is the op-level (FLA 0.3.x) convention — the newer
  *layer-level* GatedDeltaNet uses raw projection + A_log/dt_bias + in-kernel gate,
  which is different. Confirm the equivalence test passes on the actual installed
  FLA version before quoting "FLA-equivalent" in the paper. **Status 2026-07-18:
  PASSES on the RTX 5060 with fla 0.3.2 — the returned final state is the
  transposed layout (dk, dv); we transpose `initial_state` in and the result
  back out (see `_forward_fla` docstring).**
- **Joint single-set-of-weights training** (true "modality-agnostic") is the
  follow-up; the current matrix is "modality-portable" (same arch, separate
  runs). Be explicit about this in the paper.

## SGMS spike status (2026-07-18, RTX 5060)

### Surprise-gated routing — hypothesis & planned experiment (NOT yet run)

The SGMS router (`sgms/router.py`) currently routes on `z_t = W_r h_t` alone — a
plain learned linear projection (Switch-Transformer style) with **no signal for
"how well is the recurrent state predicting this token."** The `SurpriseEstimator`
in `engram/saber/saber.py` is exactly the missing quantity: it returns a clamped,
normalized per-token scalar measuring how far each token deviates from the
recurrent state's prediction.

**Hypothesis.** The first spike-gate failure (SGMS recall 0.023 vs GDR-only 0.096
@64, see table below) is partly a *weak-routing-signal* problem, not purely a
capacity problem. The literature's proposal — the recurrent state summarizes the
easy/compressible part, and the explicit memory retains only surprising tokens —
is testable in this architecture by feeding `SurpriseEstimator` output into the
router.

**Planned test (design-review rework, 2026).** The router interface now folds the
surprise signal through a **per-expert `surprise_weight`** (shape `[K]`, default
zeros ⇒ inert, backward-compatible). This replaces the original scalar-broadcast
form, which was **a measured no-op at `top_k=1`**: a per-token scalar added
identically to all K expert logits is softmax/argmax shift-invariant, so it
changed *neither* the routing decision *nor* the gate *nor* even the softmax —
only the raw `logits` tensor (feeding L_bal/L_z). The corrected per-expert form
can change the decision and is the interface for every stage below.

Sigmoid (write-strength) form: `engram/saber/saber.py`'s `SABERBackbone` already
uses `write_strength = sigmoid(γ·surprise)` for its explicit-memory write gate —
that is pre-existing SABER code, **not** the router path.

**Staged protocol (all MQAR spike: 3 seeds × 8000 steps @ T=64, laptop-class):
sequenced to avoid confounding quantity with learnability.**

1. **Stage 0 — fixed-scale probe.** *Is the signal useful at all?* Freeze the
   router, sweep per-expert `surprise_weight` (and `surprise_scale`) over a small
   grid. Not architectural learning — a cheap insurance check. If *no* setting
   moves recall, the signal/predictor design is suspect before any learnability
   work.

   **How to run** (Triton-free — the probe uses `scan_backend/delta_backend:
   reference`, so it runs without any CUDA-kernel/Triton dependency, e.g. the
   NixOS `/sbin/ldconfig` issue):
   ```
   python scripts/sgms_surprise_probe.py --config configs/sgms/surprise_probe.yaml \
       --seeds 0,1,2 --device cuda
   ```
   (If you instead want the production `associative_scan`/FLA kernels on NixOS,
   run inside `nix develop` and set `TRITON_LIBCUDA_PATH=/run/opengl-driver/lib`
   first; the reference path avoids needing this.)

   **Found (2026): the first Stage-0 run collapsed, and it was a signal-design
   flaw, not a real negative.** `w[1,-1] scale=1.0` gave `0.000±0.000` recall at
   every context on every seed. Reproduction (utilization, no GPU needed): the
   cell collapsed to `[1.000/0.000]` per layer (min_util 0.000), while `off` was
   balanced (0.465). Cause: SABER-style surprise is clamped to `[0, max]`
   (nonneg), and a nonneg surprise × per-expert weight adds a **constant
   per-token bias** that, at `scale=1.0` (vs router logits `~0.01`), sends every
   token to one expert — deterministic collapse, hence zero variance across
   seeds. **Fix: center the surprise** — `SurprisePredictor.forward` now returns
   the signed normalized `(abs_diff − mu)/(sigma+eps)` (symmetric clamp
   `[−max, max]`, mean ~ 0), so a per-expert weight modulates token-to-token
   deviation instead of a constant bias. Verified: centered `w[1,-1] scale=1`
   no longer collapses (min_util ~0.44, balanced). The probe now prints per-cell
   `min_utilization` and sweeps mixed scales; `[1,1]` remains the same-sign
   no-op control (should match `off`). Stage-0 must be re-run with the centered
   signal before drawing any conclusion.
2. **Stage 1 — learned, same `top_k` (`k=1` + straight_through).** The
   apples-to-apples comparison to the existing negative baseline (0.023 vs
   GDR-only 0.096) under the **identical `top_k=1`**. `surprise_weight` trains;
   straight-through re-attaches the gate gradient (its own bias is a known
   confound, but `top_k` stays fixed).
3. **Stage 2 — full learned (`top_k=2`).** Higher capacity. **Must include the
   control condition**: `top_k=2` with the surprise feature off (zeroed) alongside
   `top_k=2 + surprise`. If both beat the baseline, the gain is from `top_k`
   capacity, not surprise.

**Signal source (decided): lightweight standalone predictor (pattern (a)), not
the full SABER stack.** A small predictor `ĥ_t = P(h_{t-1})` over the router's
pre-norm hidden stream, with an EMA copy as the stable surprise baseline (as in
SABER's `Predictor`); surprise = normalized `|h_t − ĥ_t|`. Chosen over the
EMA-of-past-deviation ("(b)") because the scientific question is *prediction
error* (what the recurrent state fails to compress), not local volatility. `P`
sees only `h_{<t}` by construction; a **causality/leakage test** is required
(changing `h_{>t}` must not change `surprise_t`).

**Status:** router per-expert interface landed + regression tests pass (see
`tests/sgms/test_router_surprise.py`). The standalone predictor implementation
and the three stage runs are the next, separate tasks. As before: **a
hypothesis and a planned experiment, not a result** — no improvement is claimed
before a run exists.

Gate protocol: 3 seeds × 8000 steps, MQAR 8-pairs @ T=64, vocab 512, bf16,
lr 1e-3 (see `configs/sgms/spike.yaml` for the recipe note — the harder sweep
cells do not reach the MQAR "click" inside this budget on an 8 GB GPU).

| model | recall@64 | recall@4096 | min utilization | params |
|---|---|---|---|---|
| **sgms** | 0.023 ± 0.014 | 0.013 ± 0.023 | **0.184** | 4.48 M |
| B1 (fixed 3:1) | 0.039 ± 0.034 | 0.008 ± 0.000 | — | 3.16 M |
| B2 (SSD-only) | 0.000 | 0.000 | — | 3.16 M |
| B3 (GDR-only) | 0.096 ± 0.018 | 0.104 ± 0.048 | — | 3.16 M |

**Spike gate (§6.4): FAIL on quality / PASS on no-collapse.** The router does
not collapse (min utilization 0.184 ≥ 0.10), but SGMS trails GDR-only on this
pure-recall task at this optimization budget. Baseline parameter matching
(§6.2) is NOT yet honored (sgms 4.48 M vs baselines 3.16 M) — the next run
must equalize it (fewer SGMS layers or a smaller `hidden_dim` for baselines).

**Ablations run:**
- `lambda_bal = 0` (3 seeds): **R1 routing collapse confirmed empirically** —
  min utilization 0.0 on every seed (layers 1–2 go all-GDR, the output layer
  goes all-SSD or all-GDR depending on the seed), and 2/3 seeds fail to learn
  at all (acc 0.029 ± 0.050). Load balancing is load-bearing in v1.
- `shared_expert: ssd` (3 seeds): no collapse (min utilization 0.184–0.202)
  and the best SGMS result so far — one seed reaches 0.109 @64 (on par with
  B3's 0.096–0.117), though 2/3 seeds still fail to learn (acc 0.037 ± 0.063,
  params 5.79 M). The always-on SSD stabilizes utilization but does not yet
  close the reliability gap.

**C2 (specialization) first signal is positive:** on the trained spike model,
every layer shows statistically significant mutual information between expert
choice and MQAR token class (key/value/query/filler) at p = 0.002 (500
permutations; layer 1: 0.52 nats) — see `output/sgms/analysis/seed0/report.json`
(`scripts/sgms_analysis.py`).

**Next experiments:** `lambda_bal ∈ {1e-3, 1e-1}` sweep; longer runs at the
same recipe (the click was still in progress at 8k steps for SGMS); parameter
matching; passkey + state-tracking suites (tasks implemented, untrained);
**surprise-gated routing** (feed `SurpriseEstimator` → router, re-run the
spike-gate protocol in the same section).
