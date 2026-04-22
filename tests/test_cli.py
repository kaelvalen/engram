from __future__ import annotations

import logging

from prism.logging import setup_logging


def test_setup_logging_sets_debug_level():
    root = logging.getLogger("prism")
    root.handlers.clear()
    root.setLevel(logging.NOTSET)
    setup_logging("DEBUG")
    assert root.level == logging.DEBUG


def test_setup_logging_default_is_info():
    # Reset handlers so setup_logging runs fresh
    root = logging.getLogger("prism")
    root.handlers.clear()
    root.setLevel(logging.NOTSET)
    setup_logging()
    assert root.level == logging.INFO
