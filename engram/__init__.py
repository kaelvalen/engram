from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .config import ENGRAMConfig, ModalityConfig
from .logging import setup_logging

if TYPE_CHECKING:
    from .model import ENGRAMForClassification

__all__ = ["ENGRAMConfig", "ModalityConfig", "ENGRAMForClassification", "setup_logging"]


def __getattr__(name: str):
    if name == "ENGRAMForClassification":
        return importlib.import_module("engram.model").ENGRAMForClassification
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
