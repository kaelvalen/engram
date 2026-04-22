"""Optional Hugging Face–style save/load and ``PreTrainedModel`` shim."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Type

import torch

from prism.model import PRISMForClassification
from prism.training.checkpoint import cfg_from_dict, cfg_to_dict


def is_transformers_available() -> bool:
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def save_pretrained_folder(model: PRISMForClassification, save_directory: str | Path) -> None:
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
) -> PRISMForClassification:
    load_directory = Path(load_directory)
    cfg = cfg_from_dict(json.loads((load_directory / "config.json").read_text(encoding="utf-8")))
    model = PRISMForClassification(cfg)
    weights_path = load_directory / "pytorch_model.bin"
    try:
        state = torch.load(weights_path, map_location=map_location, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(state)
    return model


def get_prism_hf_classes() -> tuple[Type[Any], Type[Any]] | None:
    """Return ``(PRISMConfigHF, PRISMPreTrainedForClassification)`` or ``None`` if transformers is missing."""
    if not is_transformers_available():
        return None
    from transformers import PretrainedConfig, PreTrainedModel

    class PRISMConfigHF(PretrainedConfig):
        model_type = "prism"

        def __init__(self, prism_cfg_dict: dict[str, Any] | None = None, **kwargs: Any):
            super().__init__(**kwargs)
            self.prism_cfg_dict: dict[str, Any] = prism_cfg_dict or {}

    class PRISMPreTrainedForClassification(PreTrainedModel):
        config_class = PRISMConfigHF

        def __init__(self, config: PRISMConfigHF):
            super().__init__(config)
            pcfg = cfg_from_dict(dict(config.prism_cfg_dict))
            self.prism = PRISMForClassification(pcfg)

        def forward(self, x: torch.Tensor, modality: str, labels: torch.Tensor | None = None, **_: Any):
            return self.prism(x, modality=modality, labels=labels)

    return PRISMConfigHF, PRISMPreTrainedForClassification
