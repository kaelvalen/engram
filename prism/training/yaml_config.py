from __future__ import annotations

from pathlib import Path
from typing import Any

from prism.config import ModalityConfig, PRISMConfig

KNOWN_KEYS = {
    "modality",
    "mode",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "grad_clip",
    "hidden_dim",
    "num_heads",
    "num_layers",
    "delta_every",
    "block_pattern",
    "data_root",
    "output_dir",
    "log_level",
    "device",
    "num_workers",
    "patch_size",
    "window_size",
    "mel_bins",
    "patch_frames",
    "audio_num_classes",
    "audio_synthetic",
    "tensorboard",
    "wandb_project",
    "wandb_run_name",
    "early_stopping",
    "amp",
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
