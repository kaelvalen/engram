from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from prism.config import PRISMConfig

# PRISMConfig fields that may appear in YAML (under the 'model' section or flat).
_MODEL_KEYS = {f.name for f in fields(PRISMConfig)}
# CLI-only keys that are valid in YAML but never become PRISMConfig fields.
_CLI_ONLY_KEYS = {
    "modality",
    "mode",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "grad_clip",
    "data_root",
    "output_dir",
    "log_level",
    "device",
    "num_workers",
    "patch_size",
    "window_size",
    "ecg_multilabel",
    "ecg_task",
    "mel_bins",
    "patch_frames",
    "audio_num_classes",
    "audio_synthetic",
    "tensorboard",
    "wandb_project",
    "wandb_run_name",
    "early_stopping",
    "amp",
    "seed",
    "deterministic",
    "resume",
}
KNOWN_KEYS = _MODEL_KEYS | _CLI_ONLY_KEYS
# `layer_pattern` is a CLI alias that is translated to `block_pattern` before the
# config is built; it is intentionally NOT a PRISMConfig field.
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

    model_section = raw.get("model", {})
    if isinstance(model_section, dict):
        unknown_model = set(model_section) - KNOWN_KEYS - {"modalities"}
        if unknown_model:
            raise ValueError(f"Unknown keys in 'model' section: {sorted(unknown_model)}")

    return raw
