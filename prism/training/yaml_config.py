from __future__ import annotations

from pathlib import Path
from typing import Any

from prism.config import ModalityConfig, PRISMConfig


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("YAML root must be a mapping")
    return raw


def build_prism_config_from_mapping(data: dict[str, Any]) -> PRISMConfig:
    d = dict(data)
    modalities_raw = d.pop("modalities", None)
    modalities: list[ModalityConfig] = []
    if modalities_raw:
        for m in modalities_raw:
            modalities.append(ModalityConfig(**m))
    return PRISMConfig(**d, modalities=modalities)
