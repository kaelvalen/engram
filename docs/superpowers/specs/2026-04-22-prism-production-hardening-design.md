# PRISM Production Hardening Design

**Date:** 2026-04-22  
**Scope:** Code quality, reliability, and test coverage — foundation layer before roadmap features.  
**Approach:** Katmanlı Sağlamlaştırma (Layered Hardening) — logging first, then validation, then tests.

---

## Problem Statement

PRISM has a mathematically sound core (S4D-Complex + Gated Delta Rule) and clean module organization, but is not safe for production use due to:

- All progress and errors go to stdout via `print()` — invisible in background jobs, unredirectable
- Silent data loss in ECG loading (`except Exception: continue`)
- No checkpoint structure validation — KeyError at runtime instead of load time
- No input shape validation in model forward — cryptic tensor errors downstream
- No YAML config schema — type errors surface late
- Training loops entirely untested (0 tests for `train_epoch`, `evaluate_epoch`, CLI)
- Edge cases (batch=1, NaN input, unknown modality shape mismatch) not covered

---

## Architecture

No structural changes to the model. All changes are defensive/operational.

### Layer 1 — Logging Infrastructure

**New file:** `prism/logging.py`

```
prism/logging.py
  setup_logging(level="INFO", log_file=None)
    → configures root "prism" logger
    → console handler with level-based formatting
    → optional file handler
```

Logger hierarchy:

- `prism` — root
- `prism.data` — data loading events
- `prism.training` — epoch metrics, checkpoints
- `prism.model` — forward pass warnings

Each module gets its own logger via `logging.getLogger(__name__)`. `setup_logging()` called once in `prism/training/cli.py` entry point.

**Changes:**

- `prism/logging.py` — new
- `prism/__init__.py` — export `setup_logging`
- `prism/training/cli.py` — add `--log-level` arg, call `setup_logging()`
- `prism/training/loops.py` — replace all `print()` with `logger.info()`
- `prism/training/trainer.py` — replace all `print()` with logger calls
- `prism/data/ecg.py` — replace bare except with `logger.warning()`
- `prism/data/image.py` — replace `print()` with logger
- `prism/data/audio.py` — replace `print()` with logger
- `prism/smoke.py` — replace `print()` with logger

---

### Layer 2 — Error Handling & Validation

#### 2a. Checkpoint validation (`prism/training/checkpoint.py`)

`load_checkpoint()` validates required keys before returning:

```python
REQUIRED_KEYS = {"model_state", "cfg"}

def load_checkpoint(path, map_location=None):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    missing = REQUIRED_KEYS - set(ckpt.keys())
    if missing:
        raise ValueError(f"Checkpoint at {path!r} missing keys: {missing}")
    return ckpt
```

#### 2b. ECG data loading (`prism/data/ecg.py`)

Replace silent skip with logged warning + failure rate gate:

```python
failed = []
for ecg_id, row in df.iterrows():
    try:
        ...
    except Exception as e:
        logger.warning("Failed to load record %s: %s", ecg_id, e)
        failed.append(ecg_id)

total = len(df)
if total > 0 and len(failed) / total > 0.10:
    raise RuntimeError(
        f"ECG loading failure rate {len(failed)}/{total} exceeds 10%. "
        f"First failures: {failed[:5]}"
    )
logger.info("Loaded %d/%d ECG records", total - len(failed), total)
```

#### 2c. Model input validation (`prism/model.py`)

Add shape check at the top of `forward()`:

```python
def forward(self, x, modality: str, labels=None):
    mcfg = next((m for m in self.cfg.modalities if m.name == modality), None)
    if mcfg is None:
        raise KeyError(f"Unknown modality '{modality}'")
    if x.shape[-1] != mcfg.input_dim:
        raise ValueError(
            f"Input last dim {x.shape[-1]} does not match "
            f"expected {mcfg.input_dim} for modality '{modality}'"
        )
```

#### 2d. YAML config validation (`prism/training/yaml_config.py`)

Known CLI keys are validated against a whitelist; unknown keys raise `ValueError`:

```python
KNOWN_KEYS = {"modality", "mode", "epochs", "batch_size", "lr", "hidden_dim",
              "num_layers", "delta_every", "block_pattern", "data_root",
              "output_dir", "log_level", "tensorboard", "wandb_project",
              "early_stopping", "num_workers"}

def load_yaml_config(path):
    ...
    unknown = set(raw) - KNOWN_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {unknown}")
    return raw
```

---

### Layer 3 — Test Coverage

Target: ~25-30 new tests covering previously untested code paths.

#### `tests/test_training.py` (new)

- Single-batch `train_epoch` completes without error
- Single-batch `evaluate_epoch` returns accuracy in [0, 1]
- Trainer runs 1 epoch end-to-end (image, synthetic data)
- Checkpoint saved after epoch contains required keys

#### `tests/test_checkpoint.py` (extend existing)

- `load_checkpoint` raises `ValueError` on missing `model_state`
- `load_checkpoint` raises `ValueError` on missing `cfg`
- Valid checkpoint roundtrip (save → load → keys present)

#### `tests/test_data.py` (new)

- ECG loader logs warning and continues on single bad record
- ECG loader raises `RuntimeError` when >10% records fail
- CIFAR patchify raises if `image_size % patch_size != 0`
- Audio synthetic dataset is deterministic (same seed → same data)

#### `tests/test_model_validation.py` (new)

- `forward()` raises `ValueError` on wrong input dim
- `forward()` raises `KeyError` on unknown modality
- `forward()` handles batch size 1
- `forward()` handles sequence length 1
- NaN input propagates (output contains NaN) — confirms no silent masking

#### `tests/test_cli.py` (new)

- `--log-level DEBUG` sets logger level correctly
- YAML config with unknown key raises `ValueError`
- YAML config type override applies correctly (e.g., `epochs: 3`)

---

## Success Criteria

- `ruff check prism tests scripts train.py` passes with zero warnings
- `pytest` passes with ≥34 tests (9 existing + 25 new)
- No `print()` calls remain in `prism/` source (verified by grep)
- `logging.getLogger("prism")` hierarchy covers all submodules
- `load_checkpoint` raises `ValueError` on malformed checkpoint
- ECG loader logs warning count and raises on >10% failure rate
- Model `forward()` raises `ValueError` on shape mismatch

---

## Files Created / Modified


| File                             | Action                                         |
| -------------------------------- | ---------------------------------------------- |
| `prism/logging.py`               | Create                                         |
| `prism/__init__.py`              | Modify — export `setup_logging`                |
| `prism/model.py`                 | Modify — input shape validation                |
| `prism/training/cli.py`          | Modify — `--log-level`, call `setup_logging()` |
| `prism/training/loops.py`        | Modify — replace `print()`                     |
| `prism/training/trainer.py`      | Modify — replace `print()`                     |
| `prism/training/checkpoint.py`   | Modify — key validation                        |
| `prism/training/yaml_config.py`  | Modify — key whitelist                         |
| `prism/data/ecg.py`              | Modify — error logging + failure gate          |
| `prism/data/image.py`            | Modify — replace `print()`                     |
| `prism/data/audio.py`            | Modify — replace `print()`                     |
| `prism/smoke.py`                 | Modify — replace `print()`                     |
| `tests/test_training.py`         | Create                                         |
| `tests/test_data.py`             | Create                                         |
| `tests/test_model_validation.py` | Create                                         |
| `tests/test_cli.py`              | Create                                         |
| `tests/test_checkpoint.py`       | Extend                                         |


---

## Out of Scope

- Triton kernels, streaming decode, MoE SwiGLU (roadmap)
- Distributed training / DDP
- REST/gRPC serving layer
- Mixed precision (autocast)
- HuggingFace `PreTrainedModel` shim

