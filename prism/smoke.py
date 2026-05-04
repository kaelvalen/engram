import os as _os
import sys as _sys

# When this script is run directly (python prism/smoke.py), Python inserts the
# script's own directory (/…/prism/prism/) as sys.path[0], which causes
# `prism/logging.py` to shadow the stdlib `logging` module and break torch.
# Remove the script directory from sys.path before any other import.
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
if _sys.path and _sys.path[0] == _script_dir:
    _sys.path.pop(0)

import logging  # noqa: E402

import torch  # noqa: E402

from prism.config import ModalityConfig, PRISMConfig  # noqa: E402
from prism.logging import setup_logging  # noqa: E402
from prism.model import PRISMForClassification  # noqa: E402

setup_logging()
logger = logging.getLogger("prism.smoke")

cfg = PRISMConfig(
    hidden_dim=256,
    num_heads=8,
    num_layers=12,
    delta_every=4,
    modalities=[
        ModalityConfig(name="ecg", input_dim=12, num_classes=5),
        ModalityConfig(name="image", input_dim=48, num_classes=10),
    ],
)

model = PRISMForClassification(cfg)
logger.info("Params: %s", f"{sum(p.numel() for p in model.parameters()):,}")

B, T = 2, 128

ecg = torch.randn(B, T, 12)
labels = torch.randint(0, 5, (B,))
out = model(ecg, modality="ecg", labels=labels)
logger.info("ECG   — logits: %s, loss: %.4f", out["logits"].shape, out["loss"].item())

img = torch.randn(B, 64, 48)
labels = torch.randint(0, 10, (B,))
out = model(img, modality="image", labels=labels)
logger.info("Image — logits: %s, loss: %.4f", out["logits"].shape, out["loss"].item())
