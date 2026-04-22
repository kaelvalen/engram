from __future__ import annotations

import logging

from prism.logging import setup_logging


def test_setup_logging_sets_debug_level():
    setup_logging("DEBUG")
    logger = logging.getLogger("prism")
    assert logger.level == logging.DEBUG


def test_setup_logging_default_is_info():
    # Reset handlers so setup_logging runs fresh
    root = logging.getLogger("prism")
    root.handlers.clear()
    root.setLevel(logging.NOTSET)
    setup_logging()
    assert root.level == logging.INFO
