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
    optimizer_state: dict[str, Any] | None = None,
    scheduler_state: dict[str, Any] | None = None,
    rng_state: dict[str, Any] | None = None,
    global_step: int | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state": model_state,
        "cfg": cfg_to_dict(cfg),
    }
    if metrics is not None:
        payload["metrics"] = metrics
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    if scheduler_state is not None:
        payload["scheduler_state"] = scheduler_state
    if rng_state is not None:
        payload["rng_state"] = rng_state
    if global_step is not None:
        payload["global_step"] = global_step
    torch.save(payload, path)


_REQUIRED_KEYS = {"model_state", "cfg"}


def load_checkpoint(
    path: str | Path, map_location: str | torch.device | None = None
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    missing = _REQUIRED_KEYS - set(ckpt.keys())
    if missing:
        raise ValueError(f"Checkpoint at {str(path)!r} is missing required keys: {sorted(missing)}")
    logger.debug("Loaded checkpoint from %s (epoch=%s)", path, ckpt.get("epoch"))
    return ckpt
