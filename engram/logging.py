from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure the 'engram' logger hierarchy.

    The log level is always updated. Handlers are installed only on the first
    call; subsequent calls update the level but ignore log_file.
    """
    root = logging.getLogger("engram")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if root.handlers:
        return  # handlers already set up, just update level above

    fmt = logging.Formatter(
        "%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)
