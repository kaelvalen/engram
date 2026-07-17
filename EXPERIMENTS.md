# PRISM — Experiment Matrix

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
systematically bias PRISM *down* relative to the baseline — an apples-to-oranges
confound. Keep 1000 for any number quoted against the leaderboard (use 5000 at
500 Hz). This costs ~4× the sequence length vs the old default, but PRISM is a
linear-time model so it is O(T); correctness over speed here.

## Main table (architectures × modalities)

Shared budget: `hidden_dim=256, num_layers=12, num_heads=8` (~8M params).

| Model | Command (per seed) |
|---|---|
| PRISM hybrid (SSD + Delta 3:1) | `train.py --modality <m> --ssm-kind ssd` |
| Mamba-2 only (SSD) | `train.py --modality <m> --ssm-kind ssd --block-pattern s4` |
| Gated DeltaNet only | `train.py --modality <m> --block-pattern delta` |
| PRISM legacy (S4D + Delta) | `train.py --modality <m> --ssm-kind s4d_legacy --s4d-init lin` |
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
  FLA version before quoting "FLA-equivalent" in the paper.
- **Joint single-set-of-weights training** (true "modality-agnostic") is the
  follow-up; the current matrix is "modality-portable" (same arch, separate
  runs). Be explicit about this in the paper.
