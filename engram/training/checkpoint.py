from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import torch

from engram.config import ENGRAMConfig, ModalityConfig

logger = logging.getLogger(__name__)


def cfg_to_dict(cfg: ENGRAMConfig) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(cfg)


def cfg_from_dict(d: dict[str, Any]) -> ENGRAMConfig:
    d = dict(d)
    modalities = [ModalityConfig(**m) for m in d.pop("modalities", [])]
    return ENGRAMConfig(**d, modalities=modalities)


def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    model_state: dict[str, Any],
    cfg: ENGRAMConfig,
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


def normalize_model_state(state: dict[str, Any]) -> dict[str, Any]:
    """Remove the ``torch.compile`` wrapper prefix from legacy checkpoints."""
    prefix = "_orig_mod."
    if state and all(key.startswith(prefix) for key in state):
        return {key[len(prefix) :]: value for key, value in state.items()}
    return state


def _torch_load(path: str | Path, map_location: str | torch.device | None) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as e:
        logger.warning(
            "Checkpoint %s contains non-tensor objects; loading with weights_only=False "
            "(only for checkpoints from a trusted source): %s",
            path,
            e,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


def load_checkpoint(
    path: str | Path, map_location: str | torch.device | None = None
) -> dict[str, Any]:
    ckpt = _torch_load(path, map_location)
    missing = _REQUIRED_KEYS - set(ckpt.keys())
    if missing:
        raise ValueError(f"Checkpoint at {str(path)!r} is missing required keys: {sorted(missing)}")
    ckpt["model_state"] = normalize_model_state(ckpt["model_state"])
    logger.debug("Loaded checkpoint from %s (epoch=%s)", path, ckpt.get("epoch"))
    return ckpt
