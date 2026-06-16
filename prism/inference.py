from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from prism.config import ModalityConfig, PRISMConfig
from prism.model import PRISMForClassification

logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[PRISMForClassification, PRISMConfig, dict[str, Any]]:
    """Load a PRISM checkpoint and return the model, config, and checkpoint metadata.

    The checkpoint is expected to contain ``model_state`` and ``cfg`` (as a dict).
    Older checkpoints that store the config object directly are also supported.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    missing = {"model_state", "cfg"} - set(ckpt.keys())
    if missing:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} is missing required keys: {sorted(missing)}"
        )

    cfg_data = ckpt["cfg"]
    if isinstance(cfg_data, dict):
        d = dict(cfg_data)
        modalities = [ModalityConfig(**m) for m in d.pop("modalities", [])]
        cfg = PRISMConfig(**d, modalities=modalities)
    elif isinstance(cfg_data, PRISMConfig):
        cfg = cfg_data
    else:
        raise TypeError(f"Unexpected checkpoint cfg type: {type(cfg_data)}")

    model = PRISMForClassification(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    logger.info(
        "Loaded checkpoint from %s (epoch=%s, metrics=%s)",
        checkpoint_path,
        ckpt.get("epoch", "?"),
        ckpt.get("metrics", {}),
    )
    return model, cfg, ckpt
