from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from prism.config import ModalityConfig, PRISMConfig

logger = logging.getLogger(__name__)


def cfg_to_dict(cfg: PRISMConfig) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(cfg)


def cfg_from_dict(d: dict[str, Any]) -> PRISMConfig:
    d = dict(d)
    modalities = [ModalityConfig(**m) for m in d.pop("modalities", [])]
    return PRISMConfig(**d, modalities=modalities)


def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    model_state: dict[str, Any],
    cfg: PRISMConfig,
    metrics: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state": model_state,
        "cfg": cfg_to_dict(cfg),
    }
    if metrics:
        payload["metrics"] = metrics
    torch.save(payload, path)


_REQUIRED_KEYS = {"model_state", "cfg"}


def load_checkpoint(path: str | Path, map_location: str | torch.device | None = None) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    missing = _REQUIRED_KEYS - set(ckpt.keys())
    if missing:
        raise ValueError(f"Checkpoint at {str(path)!r} is missing required keys: {missing}")
    return ckpt
