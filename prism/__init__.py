from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .config import ModalityConfig, PRISMConfig
from .logging import setup_logging

if TYPE_CHECKING:
    from .model import PRISMForClassification

__all__ = ["PRISMConfig", "ModalityConfig", "PRISMForClassification", "setup_logging"]


def __getattr__(name: str):
    if name == "PRISMForClassification":
        return importlib.import_module("prism.model").PRISMForClassification
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
