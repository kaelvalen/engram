"""Helpers to resolve dataset roots on disk."""

from __future__ import annotations

import os


def resolve_ptbxl_root(data_root: str) -> str:
    """Return directory containing ``ptbxl_database.csv``.

    Accepts ``.../ptbxl`` or a parent directory that contains ``ptbxl/ptbxl_database.csv``
    (e.g. ``./datasets`` with ``./datasets/ptbxl``).
    """
    if os.path.isfile(os.path.join(data_root, "ptbxl_database.csv")):
        return data_root
    nested = os.path.join(data_root, "ptbxl")
    if os.path.isfile(os.path.join(nested, "ptbxl_database.csv")):
        return nested
    return nested
