from __future__ import annotations

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _serialize_numpy_state(state: tuple) -> dict[str, object]:
    name, keys, pos, has_gauss, cached_gaussian = state
    return {
        "bit_generator": name,
        "keys": keys.tolist(),
        "pos": pos,
        "has_gauss": has_gauss,
        "cached_gaussian": cached_gaussian,
    }


def _deserialize_numpy_state(state: object) -> tuple:
    if isinstance(state, dict):
        return (
            state["bit_generator"],
            np.asarray(state["keys"], dtype=np.uint32),
            state["pos"],
            state["has_gauss"],
            state["cached_gaussian"],
        )
    return state


def get_rng_state() -> dict[str, object]:
    """Capture the current RNG state for PyTorch (CPU + all CUDA devices), NumPy, and Python."""
    state: dict[str, object] = {
        "torch": torch.get_rng_state(),
        "numpy": _serialize_numpy_state(np.random.get_state()),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, object]) -> None:
    """Restore RNG states captured by :func:`get_rng_state`."""
    torch.set_rng_state(state["torch"])
    np.random.set_state(_deserialize_numpy_state(state["numpy"]))
    random.setstate(state["python"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def set_seed(seed: int | None, deterministic: bool = False) -> None:
    """Set RNG seeds for PyTorch, NumPy, Python, and (optionally) enable deterministic mode.

    Args:
        seed: Integer seed. If ``None``, no seed is set.
        deterministic: If True, set ``torch.backends.cudnn.deterministic=True`` and
            ``benchmark=False`` for fully reproducible convolutions (may be slower).
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Set random seed to %d (deterministic=%s)", seed, deterministic)
