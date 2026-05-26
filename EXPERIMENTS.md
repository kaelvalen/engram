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

## Main table (architectures × modalities)

Shared budget: `hidden_dim=256, num_layers=12, num_heads=8` (~8M params).

| Model | Command (per seed) |
|---|---|
| PRISM hybrid (SSD + Delta 3:1) | `train.py --modality <m> --ssm-kind ssd` |
| Mamba-2 only (SSD) | `train.py --modality <m> --ssm-kind ssd --block-pattern s4` |
| Gated DeltaNet only | `train.py --modality <m> --block-pattern delta` |
| PRISM legacy (S4D + Delta) | `train.py --modality <m> --ssm-kind s4d_legacy --s4d-init lin` |
| ResNet1D baseline | `scripts/train_baseline.py --model resnet1d --task ecg` |
| Transformer baseline | `scripts/train_baseline.py --model transformer --task ecg` |

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
python scripts/aggregate_results.py output/benchmarks   # mean ± std
```

Budget ≈ 30–60 single-GPU-hours (4090/A100). If constrained: drop sCIFAR,
run 2 seeds for ablations.

## Honest gaps / TODO before submission

- **All-6-tasks PTB-XL is not yet wired.** `prism/data/ecg.py` currently emits a
  single-label 5-class super-diagnostic target (argmax). The full leaderboard
  (`all/diag/sub-diag/super-diag/form/rhythm`) is **multi-label**; it needs a
  multi-hot target loader feeding `metrics.multilabel_auroc_macro`
  (`binary_auroc` already handles the per-class case). This is the first thing
  to finish before claiming the full PTB-XL table.
- **Bootstrap CIs** (1000-resample) on the test fold are not yet computed; add
  them to match Strodthoff et al. Table I before quoting "within CI".
- **FLA / mamba-ssm numbers** require a GPU; `tests/test_delta_equivalence.py`
  must pass there before trusting `--delta-backend fla`.
- **Joint single-set-of-weights training** (true "modality-agnostic") is the
  follow-up; the current matrix is "modality-portable" (same arch, separate
  runs). Be explicit about this in the paper.
