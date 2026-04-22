# PRISM — Parallel Recurrent Integrated Signal Model

> A modality-agnostic sequence architecture that processes ECG signals, images, and arbitrary continuous inputs through a single shared backbone — no modality-specific encoders, no fusion layers.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## The Idea

Modern multimodal architectures treat each modality as a separate problem — a vision encoder here, a text encoder there, a fusion layer somewhere in between. This works, but it raises a question: **if everything is a sequence of signals, why do we need different backbones?**

PRISM's answer is that we don't. A 12-lead ECG, a sequence of image patches, and a stream of sensor readings are all continuous signals sampled over time. A backbone that understands continuous-time dynamics should handle all of them — with only a lightweight per-modality input projection to adapt dimensionality.

The backbone interleaves two complementary mechanisms:

- **S4D-Complex blocks** — diagonal state-space models with input-dependent step sizes, grounded in continuous-time ODE dynamics. Good at tracking slowly evolving signal patterns across long sequences.
- **Gated Delta Rule blocks** — matrix-valued associative memory with data-dependent forget and write gates. Good at targeted recall: writing new associations and selectively overwriting stale ones.

These are interleaved 3:1 (S4 heavy), reflecting the hypothesis that continuous signal dynamics are the primary challenge, with associative memory as a supporting mechanism.

---

## Architecture

```
Input (any modality)  [B, T, input_dim]
        ↓
ModalityProjection    Linear(input_dim → hidden_dim)   ← per-modality, lightweight
        ↓
PRISMBackbone
  L0:  S4Block     ┐
  L1:  S4Block     │  S4: continuous-time signal dynamics
  L2:  S4Block     ┘
  L3:  DeltaBlock  ←  associative memory write
  L4:  S4Block     ┐
  ...              │  (repeats, delta_every=4)
  L11: DeltaBlock  ←  associative memory write
        ↓
Mean Pooling          [B, T, hidden_dim] → [B, hidden_dim]
        ↓
PerModalityHead       LayerNorm → Linear(hidden_dim → num_classes)   ← per-modality
        ↓
Output logits
```

### S4Block internals

```
x → RMSNorm → ShortCausalConv1d → S4SSM(gated, complex diagonal) → residual
  → RMSNorm → SwiGLU → residual
```

S4SSM uses a **complex diagonal A matrix** (S4D-Lin style) with **input-dependent step size** Δ_t = softplus(Linear(x_t)). The selectivity comes from Δ — the model learns how fast to evolve the state for each input token.

Discrete recurrence (Zero-Order Hold, numerically stable):
```
Ā = exp(Δ·A)
B̄ = expm1(Δ·A) / A · B
h_t = Ā ⊙ h_{t-1} + B̄ · u_t
y_t = 2·Re(C* ⊙ h_t) + D·u_t
```

### DeltaBlock internals

```
x → RMSNorm → ShortCausalConv1d → GatedDeltaRule → residual
  → RMSNorm → SwiGLU → residual
```

GatedDeltaRule maintains a **Dh×Dh matrix state per head** — a proper associative memory. Per token:
```
S_t = α_t · [S_{t-1} − β_t · (S_{t-1} k_t) k_t^T] + β_t · v_t k_t^T
o_t = S_t · q_t
```

α_t is a data-dependent forget gate (initialized near 1 for long memory). β_t controls write strength. The delta correction term `(S k) k^T` performs a targeted overwrite rather than simple accumulation.

---

## Quick Start

```bash
git clone https://github.com/kaelvalen/prism.git
cd prism
pip install -e ".[train,dev]"
```

```python
import torch
from prism import PRISMConfig, ModalityConfig, PRISMForClassification

cfg = PRISMConfig(
    hidden_dim=256,
    num_heads=8,
    num_layers=12,
    modalities=[
        ModalityConfig(name="ecg",   input_dim=12, num_classes=5),
        ModalityConfig(name="image", input_dim=48, num_classes=10),
    ]
)

model = PRISMForClassification(cfg)

# ECG: 12-lead, 128 timesteps
ecg = torch.randn(4, 128, 12)
out = model(ecg, modality="ecg", labels=torch.randint(0, 5, (4,)))
print(out["loss"].item())

# Image: 64 patches of size 4×4×3=48
img = torch.randn(4, 64, 48)
out = model(img, modality="image", labels=torch.randint(0, 10, (4,)))
print(out["loss"].item())
```

---

## Training

Canonical entrypoint is **`train.py`** at the repo root (also available as **`prism-train`** after install). Legacy wrappers `scripts/train_image.py` and `scripts/train_ecg.py` forward to the same CLI.

**Paths:** downloaded files default to **`./datasets/`** on disk. Python loaders live in the **`prism.data`** package (`prism/data/*.py`). `.gitignore` only ignores `/datasets/` and `/data/` at the **repository root**, so it cannot hide `prism/data/`.

```bash
# CIFAR-10 (torchvision auto-download under ./datasets/cifar)
python train.py --modality image --epochs 50 --batch-size 64 --lr 3e-4

# PTB-XL ECG — put files in ./datasets/ptbxl or pass a parent dir containing ptbxl/
python train.py --modality ecg --epochs 50 --batch-size 32 --lr 3e-4 --data-root ./datasets

# Synthetic mel-patch “audio” smoke / prototype (no files)
python train.py --modality audio --epochs 5 --batch-size 32

# Joint ECG + image on one shared backbone (alternating batches)
python train.py --mode joint --epochs 20 --data-root ./datasets

# Ablation: all-S4 or all-Delta blocks
python train.py --modality image --block-pattern s4 --epochs 10
python train.py --modality image --block-pattern delta --epochs 10

# Optional YAML defaults (CLI overrides)
python train.py --config configs/train.example.yaml --modality image --epochs 3

# Logging & early stopping
python train.py --modality image --tensorboard --early-stopping 5
python train.py --modality image --wandb-project prism-runs
```

Key CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--modality` | `image` | `image`, `ecg`, or `audio` (single mode) |
| `--mode` | `single` | `single` or `joint` (ecg + image) |
| `--block-pattern` | `hybrid` | `hybrid` (S4/Delta mix), `s4`, or `delta` ablation |
| `--epochs` | `50` | Training epochs |
| `--batch-size` | `64` | Batch size |
| `--lr` | `3e-4` | AdamW learning rate |
| `--hidden-dim` | `256` | Model width |
| `--num-layers` | `12` | Total blocks |
| `--data-root` | `./datasets` | **Downloaded** data root (`datasets/cifar`, `datasets/ptbxl`). Loader **code** lives in the `prism.data` package — not the same path. |
| `--tensorboard` | off | Logs under `--output-dir/tb/…` |
| `--early-stopping` | `0` | Patience on val accuracy (`0` = disabled) |

**Baselines** (1D ResNet on ECG, small Transformer on patches):

```bash
pip install -e ".[train]"
python scripts/train_baseline.py --model transformer --task image --epochs 10
python scripts/train_baseline.py --model resnet1d --task ecg --data-root ./datasets
```

**Hugging Face–style export** (folder with `config.json` + `pytorch_model.bin`; optional `transformers` subclasses via `get_prism_hf_classes()` when `[hf]` extra is installed):

```python
from prism.integrations.huggingface import save_pretrained_folder, load_pretrained_folder
save_pretrained_folder(model, "exported/prism-cifar")
model2 = load_pretrained_folder("exported/prism-cifar", map_location="cpu")
```

---

## Continuous integration

GitHub Actions workflow **`.github/workflows/ci.yml`** installs CPU PyTorch, runs **Ruff**, then **pytest**. Locally:

```bash
pip install -e ".[dev]"
ruff check prism tests scripts train.py
pytest
```

---

## Configuration

All hyperparameters live in `PRISMConfig`:

| Field | Default | Description |
|-------|---------|-------------|
| `hidden_dim` | 256 | Model dimension |
| `num_heads` | 8 | Heads in both block types |
| `num_layers` | 12 | Total blocks |
| `delta_every` | 4 | DeltaBlock every Nth layer (3:1 S4:Delta) |
| `s4_state_mult` | 2 | SSM state size = head_dim × mult |
| `s4_dt_min/max` | 0.001/0.1 | Step size range |
| `delta_chunk_size` | 64 | Chunk size for delta prefill |
| `qk_norm` | True | L2-normalize Q and K |
| `gate_bias_init` | 4.0 | σ(α) ≈ 0.98 at init (long memory) |
| `conv_kernel_size` | 4 | Short causal conv kernel |
| `ffn_expand` | 2 | SwiGLU hidden multiplier |
| `force_block_type` | `None` | Ablation: `"s4"` / `"delta"` / `None` for hybrid |

---

## Repository Layout

```
prism/
├── config.py
├── model.py
├── training/              # shared Trainer, checkpoints, CLI backing train.py
├── baselines/             # ResNet1D, small Transformer baselines
├── integrations/          # HF-style save/load; optional PreTrainedModel shim
├── modules/
├── heads/
└── data/                  # Python package (source) — not the download folder
    ├── ecg.py
    ├── image.py
    ├── audio.py           # synthetic mel patches (default) or custom .pt dumps
    └── paths.py           # PTB-XL root resolution

./datasets/                # default on-disk cache (gitignored at repo root only)

train.py                   # main training CLI
configs/train.example.yaml # optional YAML defaults
scripts/
├── train_image.py         # legacy → CLI with --modality image
├── train_ecg.py           # legacy → CLI with --modality ecg
└── train_baseline.py      # baseline training
tests/                     # pytest
```

---

## Design Decisions

**Why no modality-specific tokenizer?**
The projection layer is intentionally minimal — a single Linear per modality. The claim is that the backbone should learn the dynamics, not the tokenizer. If PRISM works, it works because S4+Delta is a good model of continuous signals, not because we engineered good features.

**Why S4D-Complex with input-dependent Δ?**
Complex diagonal A gives the model access to oscillatory dynamics — useful for periodic signals like ECG. Input-dependent Δ (borrowed from Mamba) adds selectivity: the model learns which tokens deserve more "processing time" in state space.

**Why 3:1 S4:Delta?**
Signal continuity is the primary challenge in this domain. Delta blocks are powerful but expensive (O(Dh²) state). The 3:1 ratio keeps the associative memory as a supporting mechanism without dominating the compute budget.

**Why per-modality heads, shared backbone?**
The cleanest test of the agnostic backbone claim. If two modalities can share all layers except a single linear head and still achieve competitive performance, the backbone is doing real work.

---

## Roadmap

- [ ] Triton kernel for chunked delta rule (target: 5-10× prefill speedup)
- [ ] Streaming decode with carry states (O(1) per token, O(D²) memory)
- [ ] PTB-XL training + baseline comparison (ResNet-1D, Transformer, Mamba)
- [ ] CIFAR-10 ablation: S4-only vs Delta-only vs hybrid
- [ ] MoE SwiGLU (DeepSeekMoE-style shared expert) behind config flag
- [ ] Third modality: audio (mel spectrogram patches)
- [ ] HuggingFace-compatible `PreTrainedModel` shim

---

## What This Is — and Isn't

**Is:** A clean, tested implementation of a hybrid continuous-time SSM + associative memory architecture, designed to process heterogeneous signal modalities through a single backbone. The math is correct; the architecture is grounded in the 2024–2026 frontier of efficient sequence models (S4D, GatedDeltaNet, Mamba-2).

**Isn't:** A production system or a claim that one backbone beats specialized architectures on all tasks. PRISM is a research hypothesis: *shared continuous-time dynamics are sufficient for modality-agnostic sequence modeling*. The experiments are the test.

---

*Architecture critiques, issues, and ablation results welcome.*
