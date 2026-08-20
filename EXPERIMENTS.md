# ENGRAM — Experiment Matrix

This file *locks* the experiment plan before running anything (per good ML
practice). Every table row in the paper maps to one command here. Report
**mean ± std over ≥3 seeds**, never single-best.

## Datasets

Place under `$DATA_ROOT` (default `./datasets`):

| Modality | Dataset | Where | Loader |
|---|---|---|---|
| ECG (primary) | PTB-XL 1.0.3 | physionet.org/content/ptb-xl/ → `datasets/ptbxl/` | `prism/data/ecg.py` |
| Audio (secondary) | Speech Commands v2 | torchaudio download → `datasets/audio/` | `prism/data/audio.py` |
| Vision (tertiary) | sequential CIFAR-10 | torchvision auto-download → `datasets/cifar/` | `prism/data/image.py` |

## Primary metric

PTB-XL is benchmarked with **macro one-vs-rest AUROC**, not accuracy
(Strodthoff et al. 2020). Use `prism.training.metrics.roc_auc_ovr_macro` /
`prism.training.loops.evaluate_macro_auc`. Target: match `xresnet1d101`
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
  form|rhythm|all`): multi-hot targets in `prism/data/ecg.py` via the pure,
  unit-tested `prism/data/ptbxl_tasks.py` mapping, BCEWithLogits in `model.py`,
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

## MoM spike status (2026-07-18, RTX 5060)

### Surprise-gated routing — hypothesis & planned experiment (NOT yet run)

The MoM router (`mom/router.py`) currently routes on `z_t = W_r h_t` alone — a
plain learned linear projection (Switch-Transformer style) with **no signal for
"how well is the recurrent state predicting this token."** The `SurpriseEstimator`
in `prism/saber/saber.py` is exactly the missing quantity: it returns a clamped,
normalized per-token scalar measuring how far each token deviates from the
recurrent state's prediction.

**Hypothesis.** The first spike-gate failure (MoM recall 0.023 vs GDR-only 0.096
@64, see table below) is partly a *weak-routing-signal* problem, not purely a
capacity problem. The literature's proposal — the recurrent state summarizes the
easy/compressible part, and the explicit memory retains only surprising tokens —
is testable in this architecture by feeding `SurpriseEstimator` output into the
router.

**Planned test.** Feed the surprise signal into the router as an extra per-token
feature (interface landed, see below; generation wiring is the follow-up), then
re-run the identical spike-gate protocol (3 seeds × 8000 steps, MQAR 8-pairs @
T=64, `configs/mom/spike.yaml`) and check whether it closes the gap to B3
(GDR-only). This is **a hypothesis and a planned experiment, not a result** — no
improvement is claimed before a run exists.

**Status:** the router *interface* (opts into a `[B, T]` surprise feature,
default-off `surprise_scale=0.0`) exists as of the repo-unification pass. The
`SurpriseEstimator` → router wiring at train time is the next, separate task.

Gate protocol: 3 seeds × 8000 steps, MQAR 8-pairs @ T=64, vocab 512, bf16,
lr 1e-3 (see `configs/mom/spike.yaml` for the recipe note — the harder sweep
cells do not reach the MQAR "click" inside this budget on an 8 GB GPU).

| model | recall@64 | recall@4096 | min utilization | params |
|---|---|---|---|---|
| **mom** | 0.023 ± 0.014 | 0.013 ± 0.023 | **0.184** | 4.48 M |
| B1 (fixed 3:1) | 0.039 ± 0.034 | 0.008 ± 0.000 | — | 3.16 M |
| B2 (SSD-only) | 0.000 | 0.000 | — | 3.16 M |
| B3 (GDR-only) | 0.096 ± 0.018 | 0.104 ± 0.048 | — | 3.16 M |

**Spike gate (§6.4): FAIL on quality / PASS on no-collapse.** The router does
not collapse (min utilization 0.184 ≥ 0.10), but MoM trails GDR-only on this
pure-recall task at this optimization budget. Baseline parameter matching
(§6.2) is NOT yet honored (mom 4.48 M vs baselines 3.16 M) — the next run
must equalize it (fewer MoM layers or a smaller `hidden_dim` for baselines).

**Ablations run:**
- `lambda_bal = 0` (3 seeds): **R1 routing collapse confirmed empirically** —
  min utilization 0.0 on every seed (layers 1–2 go all-GDR, the output layer
  goes all-SSD or all-GDR depending on the seed), and 2/3 seeds fail to learn
  at all (acc 0.029 ± 0.050). Load balancing is load-bearing in v1.
- `shared_expert: ssd` (3 seeds): no collapse (min utilization 0.184–0.202)
  and the best MoM result so far — one seed reaches 0.109 @64 (on par with
  B3's 0.096–0.117), though 2/3 seeds still fail to learn (acc 0.037 ± 0.063,
  params 5.79 M). The always-on SSD stabilizes utilization but does not yet
  close the reliability gap.

**C2 (specialization) first signal is positive:** on the trained spike model,
every layer shows statistically significant mutual information between expert
choice and MQAR token class (key/value/query/filler) at p = 0.002 (500
permutations; layer 1: 0.52 nats) — see `output/mom/analysis/seed0/report.json`
(`scripts/mom_analysis.py`).

**Next experiments:** `lambda_bal ∈ {1e-3, 1e-1}` sweep; longer runs at the
same recipe (the click was still in progress at 8k steps for MoM); parameter
matching; passkey + state-tracking suites (tasks implemented, untrained);
**surprise-gated routing** (feed `SurpriseEstimator` → router, re-run the
spike-gate protocol in the same section).
