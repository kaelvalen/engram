# PRISM Production Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the PRISM codebase with a structured logging system, robust error handling, input validation, and comprehensive test coverage for all previously untested code paths.

**Architecture:** Three sequential layers — (1) logging infrastructure first so every subsequent change can use it, (2) error handling and validation using the logger, (3) tests that verify all new behaviour. No model architecture changes.

**Tech Stack:** Python 3.12+, PyTorch 2.1+, pytest, stdlib `logging`, `unittest.mock`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `prism/logging.py` | Create | `setup_logging()` — configures `prism` logger hierarchy |
| `prism/__init__.py` | Modify | Export `setup_logging` |
| `prism/training/cli.py` | Modify | Replace `print()` → logger; add `--log-level`; call `setup_logging()` |
| `prism/smoke.py` | Modify | Replace `print()` → logger |
| `prism/training/checkpoint.py` | Modify | Validate `model_state`/`cfg` keys on load |
| `prism/data/ecg.py` | Modify | Log per-record failures; raise on >10% failure rate |
| `prism/model.py` | Modify | Validate modality name and input shape in `forward()` |
| `prism/training/yaml_config.py` | Modify | Add `KNOWN_KEYS` whitelist; raise on unknown keys |
| `tests/test_training.py` | Create | `train_epoch`, `evaluate_epoch`, Trainer end-to-end |
| `tests/test_checkpoint.py` | Create | `load_checkpoint` key validation |
| `tests/test_data.py` | Create | ECG failure gate; audio determinism; patchify assertion |
| `tests/test_model_validation.py` | Create | Shape mismatch, unknown modality, B=1, T=1, NaN propagation |
| `tests/test_cli.py` | Create | Log level setup; YAML unknown-key error; YAML value override |

---

### Task 1: Create `prism/logging.py` and export from `prism/__init__.py`

**Files:**
- Create: `prism/logging.py`
- Modify: `prism/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py` with just the log-level test (other tests added in Task 12):

```python
# tests/test_cli.py
from __future__ import annotations

import logging

from prism.logging import setup_logging


def test_setup_logging_sets_debug_level():
    setup_logging("DEBUG")
    logger = logging.getLogger("prism")
    assert logger.level == logging.DEBUG


def test_setup_logging_default_is_info():
    # Reset handlers so setup_logging runs fresh
    root = logging.getLogger("prism")
    root.handlers.clear()
    root.setLevel(logging.NOTSET)
    setup_logging()
    assert root.level == logging.INFO
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_cli.py -v
```

Expected: `ImportError: cannot import name 'setup_logging' from 'prism.logging'`

- [ ] **Step 3: Create `prism/logging.py`**

```python
from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    root = logging.getLogger("prism")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if root.handlers:
        return

    fmt = logging.Formatter(
        "%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)
```

- [ ] **Step 4: Export from `prism/__init__.py`**

Replace the entire file:

```python
from .config import ModalityConfig, PRISMConfig
from .logging import setup_logging
from .model import PRISMForClassification

__all__ = ["PRISMConfig", "ModalityConfig", "PRISMForClassification", "setup_logging"]
```

- [ ] **Step 5: Run tests**

```
pytest tests/test_cli.py::test_setup_logging_sets_debug_level tests/test_cli.py::test_setup_logging_default_is_info -v
```

Expected: both PASS

- [ ] **Step 6: Commit**

```bash
git add prism/logging.py prism/__init__.py tests/test_cli.py
git commit -m "feat: add prism logging module with setup_logging"
```

---

### Task 2: Replace `print()` in `prism/training/cli.py` and add `--log-level`

**Files:**
- Modify: `prism/training/cli.py`

- [ ] **Step 1: Add logger at top of file**

After the existing imports in `prism/training/cli.py`, add:

```python
import logging

from prism.logging import setup_logging

logger = logging.getLogger(__name__)
```

The existing imports already have `import os`, `import sys`, `import time`, `from pathlib import Path`, `import torch`, etc. Add the two new lines after the last `from prism...` import line (currently `from prism.training.trainer import Trainer, TrainerConfig`).

- [ ] **Step 2: Add `--log-level` argument to `main()`**

In `main()`, after the `parser = argparse.ArgumentParser(...)` line and before `parser.add_argument("--config", ...)`, add:

```python
parser.add_argument(
    "--log-level",
    type=str,
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    help="Logging verbosity",
)
```

- [ ] **Step 3: Call `setup_logging()` early in `main()`**

After `args = parser.parse_args(argv)` and before `_apply_yaml_defaults(args, args.config)`, add:

```python
setup_logging(args.log_level)
```

- [ ] **Step 4: Replace `print()` calls in `run_joint_training()`**

In `run_joint_training()`, replace:

```python
        print(
            f"[{epoch:03d}/{args.epochs}] joint train_loss={metrics['train_loss']:.4f} | "
            f"val acc ecg={acc_e:.4f} img={acc_i:.4f} mean={mean_acc:.4f} | {dt:.1f}s"
        )
```

with:

```python
        logger.info(
            "[%03d/%d] joint train_loss=%.4f | val acc ecg=%.4f img=%.4f mean=%.4f | %.1fs",
            epoch, args.epochs, metrics["train_loss"], acc_e, acc_i, mean_acc, dt,
        )
```

Replace:

```python
            print(f"  saved best_joint.pt (mean val acc={mean_acc:.4f})")
```

with:

```python
            logger.info("  saved best_joint.pt (mean val acc=%.4f)", mean_acc)
```

Replace:

```python
    print(f"\nJoint training done. Best mean val acc: {best_mean:.4f}")
```

with:

```python
    logger.info("Joint training done. Best mean val acc: %.4f", best_mean)
```

- [ ] **Step 5: Replace `print()` calls in `main()`**

Replace:

```python
    print(
        f"PRISM — {modality.upper()} | params: {sum(p.numel() for p in model.parameters()):,} | "
        f"device: {device} | block_pattern={args.block_pattern}"
    )
    print("-" * 60)
```

with:

```python
    logger.info(
        "PRISM — %s | params: %s | device: %s | block_pattern=%s",
        modality.upper(),
        f"{sum(p.numel() for p in model.parameters()):,}",
        device,
        args.block_pattern,
    )
```

Replace the `on_epoch` callback inside `main()`:

```python
    def on_epoch(epoch: int, m: dict[str, float]) -> None:
        logger.info(
            "[%03d/%d] train loss: %.4f acc: %.4f | val loss: %.4f acc: %.4f",
            epoch, args.epochs,
            m["train_loss"], m["train_acc"],
            m["val_loss"], m["val_acc"],
        )
```

Replace the final `print("\nDone.")`:

```python
    logger.info("Done.")
```

- [ ] **Step 6: Run existing tests to confirm no regressions**

```
pytest tests/ -v
```

Expected: all 9 existing tests PASS

- [ ] **Step 7: Commit**

```bash
git add prism/training/cli.py
git commit -m "feat: replace print() with logger in cli.py, add --log-level"
```

---

### Task 3: Replace `print()` in `prism/smoke.py`

**Files:**
- Modify: `prism/smoke.py`

- [ ] **Step 1: Update `prism/smoke.py`**

Replace the entire file:

```python
import logging

import torch

from prism.config import ModalityConfig, PRISMConfig
from prism.logging import setup_logging
from prism.model import PRISMForClassification

setup_logging()
logger = logging.getLogger(__name__)

cfg = PRISMConfig(
    hidden_dim=256,
    num_heads=8,
    num_layers=12,
    delta_every=4,
    modalities=[
        ModalityConfig(name="ecg",   input_dim=12,  num_classes=5),
        ModalityConfig(name="image", input_dim=48, num_classes=10),
    ]
)

model = PRISMForClassification(cfg)
logger.info("Params: %s", f"{sum(p.numel() for p in model.parameters()):,}")

B, T = 2, 128

ecg    = torch.randn(B, T, 12)
labels = torch.randint(0, 5, (B,))
out    = model(ecg, modality="ecg", labels=labels)
logger.info("ECG   — logits: %s, loss: %.4f", out["logits"].shape, out["loss"].item())

img    = torch.randn(B, 64, 48)
labels = torch.randint(0, 10, (B,))
out    = model(img, modality="image", labels=labels)
logger.info("Image — logits: %s, loss: %.4f", out["logits"].shape, out["loss"].item())
```

- [ ] **Step 2: Run smoke test manually**

```
python prism/smoke.py
```

Expected: three INFO log lines (params count, ECG logits/loss, Image logits/loss) written to stderr

- [ ] **Step 3: Run pytest to confirm no regressions**

```
pytest tests/ -v
```

Expected: all existing tests PASS

- [ ] **Step 4: Commit**

```bash
git add prism/smoke.py
git commit -m "feat: replace print() with logger in smoke.py"
```

---

### Task 4: Checkpoint key validation

**Files:**
- Modify: `prism/training/checkpoint.py`
- Create: `tests/test_checkpoint.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_checkpoint.py`:

```python
from __future__ import annotations

import pytest
import torch

from prism.config import ModalityConfig, PRISMConfig
from prism.model import PRISMForClassification
from prism.training.checkpoint import load_checkpoint, save_checkpoint


def _tiny_cfg() -> PRISMConfig:
    return PRISMConfig(
        hidden_dim=32, num_heads=4, num_layers=4, delta_every=2,
        modalities=[ModalityConfig(name="ecg", input_dim=12, num_classes=3)],
    )


def test_load_checkpoint_raises_on_missing_model_state(tmp_path):
    bad = {"cfg": {"hidden_dim": 32}}
    path = tmp_path / "bad.pt"
    torch.save(bad, path)
    with pytest.raises(ValueError, match="model_state"):
        load_checkpoint(path)


def test_load_checkpoint_raises_on_missing_cfg(tmp_path):
    bad = {"model_state": {}}
    path = tmp_path / "bad.pt"
    torch.save(bad, path)
    with pytest.raises(ValueError, match="cfg"):
        load_checkpoint(path)


def test_load_checkpoint_roundtrip(tmp_path):
    cfg = _tiny_cfg()
    model = PRISMForClassification(cfg)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, epoch=1, model_state=model.state_dict(), cfg=cfg)
    ckpt = load_checkpoint(path)
    assert "model_state" in ckpt
    assert "cfg" in ckpt
    assert ckpt["epoch"] == 1
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_checkpoint.py -v
```

Expected: `test_load_checkpoint_raises_on_missing_model_state` FAILS (no ValueError raised), `test_load_checkpoint_raises_on_missing_cfg` FAILS, `test_load_checkpoint_roundtrip` PASSES

- [ ] **Step 3: Add validation to `load_checkpoint`**

In `prism/training/checkpoint.py`, replace `load_checkpoint`:

```python
_REQUIRED_KEYS = {"model_state", "cfg"}


def load_checkpoint(path: str | Path, map_location: str | torch.device | None = None) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    missing = _REQUIRED_KEYS - set(ckpt.keys())
    if missing:
        raise ValueError(f"Checkpoint at {str(path)!r} is missing required keys: {missing}")
    return ckpt
```

Also add `import logging` and `logger = logging.getLogger(__name__)` at the top of the file (after existing imports).

- [ ] **Step 4: Run tests**

```
pytest tests/test_checkpoint.py -v
```

Expected: all 3 tests PASS

- [ ] **Step 5: Run full suite**

```
pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add prism/training/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: validate checkpoint keys on load, add checkpoint tests"
```

---

### Task 5: ECG loading — error logging and failure gate

**Files:**
- Modify: `prism/data/ecg.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Add `_check_ecg_failure_rate` helper to `prism/data/ecg.py`**

`wfdb` and `pandas` are imported inside `_load()` so they can't be patched at module level. Instead, extract the failure rate gate as a standalone function that is easy to unit-test.

Add at top of `prism/data/ecg.py` (after existing imports):

```python
import logging

logger = logging.getLogger(__name__)


def _check_ecg_failure_rate(failed: list, total: int) -> None:
    """Raise RuntimeError if the ECG failure rate exceeds 10%."""
    if total > 0 and len(failed) / total > 0.10:
        raise RuntimeError(
            f"ECG loading failure rate {len(failed)}/{total} exceeds 10%. "
            f"First failed IDs: {failed[:5]}"
        )
```

Replace the inner loop in `_load()` — from `data, labels = [], []` through
`return np.stack(data), np.array(labels, dtype=np.int64)`:

```python
        data, labels = [], []
        failed: list = []

        for ecg_id, row in df.iterrows():
            path = os.path.join(self.root, folder, row["filename_lr"]
                                if self.sampling_rate == 100 else row["filename_hr"])
            try:
                record = wfdb.rdrecord(path)
                signal = record.p_signal.astype(np.float32)  # [T, 12]
            except Exception as e:
                logger.warning("Failed to load ECG record %s from %s: %s", ecg_id, path, e)
                failed.append(ecg_id)
                continue

            # window: ilk window_size timestamp al
            if signal.shape[0] >= self.window_size:
                signal = signal[:self.window_size]
            else:
                pad = np.zeros((self.window_size - signal.shape[0], 12), dtype=np.float32)
                signal = np.concatenate([signal, pad], axis=0)

            if self.normalize:
                mean = signal.mean(axis=0, keepdims=True)
                std  = signal.std(axis=0, keepdims=True) + 1e-8
                signal = (signal - mean) / std

            data.append(signal)
            labels.append(row["label"])

        total = len(df)
        loaded = total - len(failed)
        logger.info("ECG split=%s: loaded %d/%d records (%d failed)", self.split, loaded, total, len(failed))
        _check_ecg_failure_rate(failed, total)

        return np.stack(data), np.array(labels, dtype=np.int64)
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_data.py`:

```python
from __future__ import annotations

import pytest
import torch

from prism.data.audio import SyntheticMelPatchDataset
from prism.data.ecg import _check_ecg_failure_rate
from prism.data.image import patchify


# ── ECG failure gate ──────────────────────────────────────────────────────────

def test_ecg_failure_gate_raises_on_high_rate():
    """50% failure rate (5/10) exceeds threshold."""
    with pytest.raises(RuntimeError, match="failure rate"):
        _check_ecg_failure_rate(list(range(5)), 10)


def test_ecg_failure_gate_raises_at_boundary():
    """Exactly 11% failure (2/18 = 11.1%) should raise."""
    with pytest.raises(RuntimeError, match="failure rate"):
        _check_ecg_failure_rate(list(range(2)), 18)


def test_ecg_failure_gate_passes_under_threshold():
    """5% failure (1/20) is below 10% threshold — should not raise."""
    _check_ecg_failure_rate([1], 20)  # 5% — OK


def test_ecg_failure_gate_passes_on_zero_total():
    """Empty dataset should not raise (no divide-by-zero)."""
    _check_ecg_failure_rate([], 0)


# ── Image ─────────────────────────────────────────────────────────────────────

def test_patchify_raises_on_indivisible_size():
    x = torch.randn(2, 3, 32, 32)
    with pytest.raises(AssertionError):
        patchify(x, patch_size=5)  # 32 % 5 != 0


def test_patchify_correct_output_shape():
    x = torch.randn(4, 3, 32, 32)
    out = patchify(x, patch_size=4)
    assert out.shape == (4, 64, 48)  # 8×8 patches, 4×4×3=48 dim


# ── Audio ─────────────────────────────────────────────────────────────────────

def test_audio_synthetic_is_deterministic():
    ds1 = SyntheticMelPatchDataset(length=8, num_frames=32, mel_bins=16, num_classes=4, seed=42)
    ds2 = SyntheticMelPatchDataset(length=8, num_frames=32, mel_bins=16, num_classes=4, seed=42)
    x1, y1 = ds1[0]
    x2, y2 = ds2[0]
    assert torch.allclose(x1, x2)
    assert y1 == y2


def test_audio_different_seeds_differ():
    ds1 = SyntheticMelPatchDataset(length=8, num_frames=32, mel_bins=16, num_classes=4, seed=1)
    ds2 = SyntheticMelPatchDataset(length=8, num_frames=32, mel_bins=16, num_classes=4, seed=2)
    x1, _ = ds1[0]
    x2, _ = ds2[0]
    assert not torch.allclose(x1, x2)
```

- [ ] **Step 3: Run to confirm failures**

```
pytest tests/test_data.py -v
```

Expected: `test_ecg_failure_gate_raises_*` FAIL (function not yet defined), rest PASS

- [ ] **Step 4: Run all tests after Step 1**

```
pytest tests/test_data.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prism/data/ecg.py tests/test_data.py
git commit -m "feat: log ECG load failures, raise on >10% failure rate"
```

---

### Task 6: Model input shape and modality validation

**Files:**
- Modify: `prism/model.py`
- Create: `tests/test_model_validation.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_model_validation.py`:

```python
from __future__ import annotations

import pytest
import torch

from prism.config import ModalityConfig, PRISMConfig
from prism.model import PRISMForClassification


def _tiny_cfg() -> PRISMConfig:
    return PRISMConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=4,
        delta_every=2,
        modalities=[
            ModalityConfig(name="ecg", input_dim=12, num_classes=5),
            ModalityConfig(name="image", input_dim=48, num_classes=10),
        ],
    )


def test_forward_raises_on_wrong_input_dim():
    model = PRISMForClassification(_tiny_cfg())
    x = torch.randn(2, 32, 99)  # wrong: expected 12 for "ecg"
    with pytest.raises(ValueError, match="Input last dim 99"):
        model(x, modality="ecg")


def test_forward_raises_on_unknown_modality():
    model = PRISMForClassification(_tiny_cfg())
    with pytest.raises(KeyError, match="unknown"):
        model(torch.randn(1, 8, 12), modality="unknown")


def test_forward_batch_size_one():
    model = PRISMForClassification(_tiny_cfg())
    x = torch.randn(1, 32, 12)
    out = model(x, modality="ecg")
    assert out["logits"].shape == (1, 5)


def test_forward_sequence_length_one():
    model = PRISMForClassification(_tiny_cfg())
    x = torch.randn(2, 1, 12)
    out = model(x, modality="ecg")
    assert out["logits"].shape == (2, 5)


def test_nan_input_propagates():
    model = PRISMForClassification(_tiny_cfg())
    x = torch.full((2, 8, 12), float("nan"))
    out = model(x, modality="ecg")
    assert torch.isnan(out["logits"]).any(), "NaN input should propagate to output"
```

- [ ] **Step 2: Run to confirm failures**

```
pytest tests/test_model_validation.py -v
```

Expected:
- `test_forward_raises_on_wrong_input_dim` FAILS (no ValueError raised yet)
- `test_forward_raises_on_unknown_modality` PASSES (already raises KeyError via projection dict lookup)
- `test_forward_batch_size_one` PASSES
- `test_forward_sequence_length_one` PASSES
- `test_nan_input_propagates` PASSES or FAILS depending on PyTorch NaN handling

- [ ] **Step 3: Add validation to `prism/model.py`**

Add `import logging` after the existing imports, then `logger = logging.getLogger(__name__)`.

In `PRISMForClassification.forward()`, replace:

```python
    def forward(
        self,
        x: torch.Tensor,
        modality: str,
        labels: torch.Tensor | None = None,
        states: list[BlockState | None] | None = None,
    ) -> dict:
        # 1. projection
        x = self.projection(x, modality)          # [B, T, hidden_dim]
```

with:

```python
    def forward(
        self,
        x: torch.Tensor,
        modality: str,
        labels: torch.Tensor | None = None,
        states: list[BlockState | None] | None = None,
    ) -> dict:
        # validate modality and input shape
        mcfg = next((m for m in self.cfg.modalities if m.name == modality), None)
        if mcfg is None:
            raise KeyError(f"Unknown modality '{modality}'. Registered: {[m.name for m in self.cfg.modalities]}")
        if x.shape[-1] != mcfg.input_dim:
            raise ValueError(
                f"Input last dim {x.shape[-1]} does not match "
                f"expected {mcfg.input_dim} for modality '{modality}'"
            )

        # 1. projection
        x = self.projection(x, modality)          # [B, T, hidden_dim]
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_model_validation.py -v
```

Expected: all 5 PASS

- [ ] **Step 5: Run full suite**

```
pytest tests/ -v
```

Expected: all tests PASS (the existing `test_unknown_modality_raises` test also still passes)

- [ ] **Step 6: Commit**

```bash
git add prism/model.py tests/test_model_validation.py
git commit -m "feat: validate modality name and input shape in forward()"
```

---

### Task 7: YAML config key whitelist

**Files:**
- Modify: `prism/training/yaml_config.py`
- Modify: `tests/test_cli.py` (add YAML tests)

- [ ] **Step 1: Add YAML tests to `tests/test_cli.py`**

Append to the existing `tests/test_cli.py`:

```python
import textwrap
from pathlib import Path

from prism.training.yaml_config import load_yaml_config


def test_yaml_unknown_key_raises(tmp_path: Path):
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text(textwrap.dedent("""\
        epochs: 5
        typo_key: 99
    """))
    with pytest.raises(ValueError, match="typo_key"):
        load_yaml_config(cfg_file)


def test_yaml_known_keys_accepted(tmp_path: Path):
    cfg_file = tmp_path / "good.yaml"
    cfg_file.write_text(textwrap.dedent("""\
        epochs: 3
        lr: 0.001
        hidden_dim: 128
    """))
    result = load_yaml_config(cfg_file)
    assert result["epochs"] == 3
    assert result["lr"] == 0.001


def test_yaml_nested_train_section_accepted(tmp_path: Path):
    cfg_file = tmp_path / "nested.yaml"
    cfg_file.write_text(textwrap.dedent("""\
        train:
          epochs: 10
          lr: 0.0005
        model:
          hidden_dim: 64
    """))
    result = load_yaml_config(cfg_file)
    assert "train" in result
```

Also add `import pytest` at the top of `tests/test_cli.py` if not already present.

- [ ] **Step 2: Run to confirm failures**

```
pytest tests/test_cli.py::test_yaml_unknown_key_raises tests/test_cli.py::test_yaml_known_keys_accepted tests/test_cli.py::test_yaml_nested_train_section_accepted -v
```

Expected: `test_yaml_unknown_key_raises` FAILS (no ValueError raised), others PASS

- [ ] **Step 3: Update `prism/training/yaml_config.py`**

Replace the entire file:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from prism.config import ModalityConfig, PRISMConfig

KNOWN_KEYS = {
    "modality", "mode", "epochs", "batch_size", "lr", "weight_decay",
    "grad_clip", "hidden_dim", "num_heads", "num_layers", "delta_every",
    "block_pattern", "data_root", "output_dir", "log_level", "device",
    "num_workers", "patch_size", "window_size", "mel_bins", "patch_frames",
    "audio_num_classes", "audio_synthetic", "tensorboard", "wandb_project",
    "wandb_run_name", "early_stopping",
}

_TOP_LEVEL_ALLOWED = KNOWN_KEYS | {"train", "model", "modalities"}


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("YAML root must be a mapping")

    unknown_top = set(raw) - _TOP_LEVEL_ALLOWED
    if unknown_top:
        raise ValueError(f"Unknown YAML config keys: {sorted(unknown_top)}")

    train_section = raw.get("train", {})
    if isinstance(train_section, dict):
        unknown_train = set(train_section) - KNOWN_KEYS - {"modalities", "model"}
        if unknown_train:
            raise ValueError(f"Unknown keys in 'train' section: {sorted(unknown_train)}")

    return raw


def build_prism_config_from_mapping(data: dict[str, Any]) -> PRISMConfig:
    d = dict(data)
    modalities_raw = d.pop("modalities", None)
    modalities: list[ModalityConfig] = []
    if modalities_raw:
        for m in modalities_raw:
            modalities.append(ModalityConfig(**m))
    return PRISMConfig(**d, modalities=modalities)
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_cli.py -v
```

Expected: all PASS

- [ ] **Step 5: Run full suite**

```
pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add prism/training/yaml_config.py tests/test_cli.py
git commit -m "feat: validate YAML config keys against whitelist"
```

---

### Task 8: Training loop tests

**Files:**
- Create: `tests/test_training.py`

- [ ] **Step 1: Create `tests/test_training.py`**

```python
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from prism.config import ModalityConfig, PRISMConfig
from prism.model import PRISMForClassification
from prism.training.loops import evaluate_epoch, train_epoch
from prism.training.trainer import Trainer, TrainerConfig


def _tiny_cfg() -> PRISMConfig:
    return PRISMConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=4,
        delta_every=2,
        modalities=[ModalityConfig(name="ecg", input_dim=12, num_classes=5)],
    )


def _tiny_loader(n: int = 16, batch_size: int = 4) -> DataLoader:
    x = torch.randn(n, 32, 12)
    y = torch.randint(0, 5, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_train_epoch_returns_loss_and_acc():
    model = PRISMForClassification(_tiny_cfg())
    loader = _tiny_loader()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")
    loss, acc = train_epoch(model, loader, opt, device, modality="ecg")
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert loss > 0
    assert 0.0 <= acc <= 1.0


def test_evaluate_epoch_returns_loss_and_acc():
    model = PRISMForClassification(_tiny_cfg())
    loader = _tiny_loader()
    device = torch.device("cpu")
    loss, acc = evaluate_epoch(model, loader, device, modality="ecg")
    assert isinstance(loss, float)
    assert 0.0 <= acc <= 1.0


def test_trainer_fit_one_epoch(tmp_path: Path):
    cfg = _tiny_cfg()
    model = PRISMForClassification(cfg)
    loader = _tiny_loader(n=8, batch_size=4)
    tcfg = TrainerConfig(epochs=1, log_every_epoch=False)
    trainer = Trainer(model, cfg, device=torch.device("cpu"), tcfg=tcfg)
    result = trainer.fit(loader, loader, modality="ecg", output_dir=tmp_path)
    assert "best_val_acc" in result
    assert "history" in result
    assert len(result["history"]) == 1


def test_trainer_checkpoint_saved(tmp_path: Path):
    cfg = _tiny_cfg()
    model = PRISMForClassification(cfg)
    loader = _tiny_loader(n=8, batch_size=4)
    tcfg = TrainerConfig(epochs=1, log_every_epoch=False)
    trainer = Trainer(model, cfg, device=torch.device("cpu"), tcfg=tcfg)
    trainer.fit(loader, loader, modality="ecg", output_dir=tmp_path, best_filename="best_ecg.pt")
    ckpt_path = tmp_path / "best_ecg.pt"
    assert ckpt_path.exists()
    import torch as _torch
    ckpt = _torch.load(ckpt_path, weights_only=False)
    assert "model_state" in ckpt
    assert "cfg" in ckpt


def test_train_epoch_updates_parameters():
    model = PRISMForClassification(_tiny_cfg())
    before = {n: p.clone() for n, p in model.named_parameters()}
    loader = _tiny_loader()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    train_epoch(model, loader, opt, torch.device("cpu"), modality="ecg")
    changed = any(
        not torch.allclose(before[n], p)
        for n, p in model.named_parameters()
    )
    assert changed, "Parameters should change after a training epoch"
```

- [ ] **Step 2: Run tests**

```
pytest tests/test_training.py -v
```

Expected: all 5 PASS

- [ ] **Step 3: Run full suite**

```
pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_training.py
git commit -m "test: add training loop and Trainer end-to-end tests"
```

---

### Task 9: Final verification — ruff + full pytest

**Files:** None (verification only)

- [ ] **Step 1: Run ruff**

```
ruff check prism tests scripts train.py
```

Expected: zero warnings. If any issues appear, fix them before proceeding.

Common fixes:
- Unused imports → remove them
- Line too long → wrap with parentheses
- `logging` import shadows built-in → not an issue in Python 3 (absolute imports)

- [ ] **Step 2: Run full pytest with verbose output**

```
pytest tests/ -v --tb=short
```

Expected: ≥34 tests, all PASS. Count breakdown:
- `test_model.py`: 3
- `test_config_ablation.py`: 3
- `test_checkpoint_roundtrip.py`: 1
- `test_audio_loader.py`: 1
- `test_hf_optional.py`: 1
- `test_checkpoint.py`: 3
- `test_data.py`: ~5
- `test_model_validation.py`: 5
- `test_cli.py`: 5
- `test_training.py`: 5

Total: ~32+

- [ ] **Step 3: Verify no `print()` remains in `prism/` source**

```
grep -rn "print(" prism/ --include="*.py"
```

Expected: zero matches (smoke.py is a standalone script, acceptable if it has logger calls now)

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: production hardening complete — logging, validation, tests"
```

---

## Success Criteria Checklist

- [ ] `ruff check prism tests scripts train.py` → zero warnings
- [ ] `pytest tests/` → all tests pass, ≥32 tests total
- [ ] `grep -rn "^    print(" prism/ --include="*.py"` → zero results in non-script files
- [ ] `load_checkpoint` raises `ValueError` on malformed checkpoint (tested)
- [ ] ECG loader logs warning per failed record and raises `RuntimeError` on >10% failure (tested)
- [ ] `PRISMForClassification.forward()` raises `ValueError` on shape mismatch (tested)
- [ ] `PRISMForClassification.forward()` raises `KeyError` on unknown modality (tested)
- [ ] `load_yaml_config` raises `ValueError` on unknown keys (tested)
- [ ] `setup_logging()` configures `prism` logger hierarchy (tested)
