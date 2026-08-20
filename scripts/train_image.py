"""Legacy entrypoint: forwards to ``engram.training.cli`` with ``--modality image``."""

from __future__ import annotations

import sys

from engram.training.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--modality" not in argv:
        argv = ["--modality", "image", *argv]
    main(argv)
