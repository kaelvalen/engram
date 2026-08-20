from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import pytest
from engram.logging import setup_logging
from engram.training.yaml_config import load_yaml_config


def test_setup_logging_sets_debug_level():
    root = logging.getLogger("engram")
    root.handlers.clear()
    root.setLevel(logging.NOTSET)
    setup_logging("DEBUG")
    assert root.level == logging.DEBUG


def test_setup_logging_default_is_info():
    # Reset handlers so setup_logging runs fresh
    root = logging.getLogger("engram")
    root.handlers.clear()
    root.setLevel(logging.NOTSET)
    setup_logging()
    assert root.level == logging.INFO


def test_yaml_unknown_key_raises(tmp_path: Path):
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text(
        textwrap.dedent("""\
        epochs: 5
        typo_key: 99
    """)
    )
    with pytest.raises(ValueError, match="typo_key"):
        load_yaml_config(cfg_file)


def test_yaml_known_keys_accepted(tmp_path: Path):
    cfg_file = tmp_path / "good.yaml"
    cfg_file.write_text(
        textwrap.dedent("""\
        epochs: 3
        lr: 0.001
        hidden_dim: 128
    """)
    )
    result = load_yaml_config(cfg_file)
    assert result["epochs"] == 3
    assert result["lr"] == 0.001


def test_yaml_nested_train_section_accepted(tmp_path: Path):
    cfg_file = tmp_path / "nested.yaml"
    cfg_file.write_text(
        textwrap.dedent("""\
        train:
          epochs: 10
          lr: 0.0005
        model:
          hidden_dim: 64
    """)
    )
    result = load_yaml_config(cfg_file)
    assert "train" in result
