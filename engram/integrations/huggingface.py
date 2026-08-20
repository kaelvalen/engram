"""Optional Hugging Face–style save/load and ``PreTrainedModel`` shim."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Type

import torch

from engram.model import ENGRAMForClassification
from engram.training.checkpoint import cfg_from_dict, cfg_to_dict


def is_transformers_available() -> bool:
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def save_pretrained_folder(model: ENGRAMForClassification, save_directory: str | Path) -> None:
    """Write ``config.json`` + ``pytorch_model.bin`` (layout compatible with HF tooling)."""
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)
    cfg_path = save_directory / "config.json"
    cfg_path.write_text(json.dumps(cfg_to_dict(model.cfg), indent=2), encoding="utf-8")
    torch.save(model.state_dict(), save_directory / "pytorch_model.bin")


def load_pretrained_folder(
    load_directory: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> ENGRAMForClassification:
    load_directory = Path(load_directory)
    cfg = cfg_from_dict(json.loads((load_directory / "config.json").read_text(encoding="utf-8")))
    model = ENGRAMForClassification(cfg)
    weights_path = load_directory / "pytorch_model.bin"
    try:
        state = torch.load(weights_path, map_location=map_location, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(state)
    return model


def get_engram_hf_classes() -> tuple[Type[Any], Type[Any]] | None:
    """Return ``(ENGRAMConfigHF, ENGRAMPreTrainedForClassification)`` or ``None`` if transformers is missing."""
    if not is_transformers_available():
        return None
    from transformers import PretrainedConfig, PreTrainedModel

    class ENGRAMConfigHF(PretrainedConfig):
        model_type = "engram"

        def __init__(self, engram_cfg_dict: dict[str, Any] | None = None, **kwargs: Any):
            super().__init__(**kwargs)
            self.engram_cfg_dict: dict[str, Any] = engram_cfg_dict or {}

    class ENGRAMPreTrainedForClassification(PreTrainedModel):
        config_class = ENGRAMConfigHF

        def __init__(self, config: ENGRAMConfigHF):
            super().__init__(config)
            pcfg = cfg_from_dict(dict(config.engram_cfg_dict))
            self.engram = ENGRAMForClassification(pcfg)

        def forward(
            self, x: torch.Tensor, modality: str, labels: torch.Tensor | None = None, **_: Any
        ):
            return self.engram(x, modality=modality, labels=labels)

    return ENGRAMConfigHF, ENGRAMPreTrainedForClassification
