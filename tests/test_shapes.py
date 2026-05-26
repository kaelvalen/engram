"""Shape contracts: every block is (B, L, D) -> (B, L, D); every modality
projection and head produces the right dimensions, with no silent broadcasting.
"""

from __future__ import annotations

import pytest
import torch
from prism.config import ModalityConfig, PRISMConfig
from prism.model import PRISMForClassification
from prism.modules.attention import SWABlock
from prism.modules.delta import DeltaBlock
from prism.modules.s4 import S4Block
from prism.modules.ssd import SSDBlock

BLOCKS = {
    "ssd": lambda: SSDBlock(hidden_dim=32, num_heads=4, state_dim=16),
    "s4d": lambda: S4Block(hidden_dim=32, num_heads=4),
    "delta": lambda: DeltaBlock(hidden_dim=32, num_heads=4),
    "swa": lambda: SWABlock(hidden_dim=32, num_heads=4, window=8),
}


@pytest.mark.parametrize("name", list(BLOCKS))
@pytest.mark.parametrize("B,L", [(1, 1), (2, 7), (3, 64)])
def test_block_preserves_shape(name, B, L):
    torch.manual_seed(0)
    blk = BLOCKS[name]()
    x = torch.randn(B, L, 32)
    out = blk(x)
    y = out[0]
    assert y.shape == (B, L, 32), f"{name} changed shape: {tuple(y.shape)}"
    assert torch.isfinite(y).all()


def test_modality_projection_and_heads():
    cfg = PRISMConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=4,
        modalities=[
            ModalityConfig("ecg", 12, 5),
            ModalityConfig("image", 48, 10),
            ModalityConfig("audio", 64, 35),
        ],
    )
    model = PRISMForClassification(cfg)
    for name, in_dim, n_cls in [("ecg", 12, 5), ("image", 48, 10), ("audio", 64, 35)]:
        x = torch.randn(2, 16, in_dim)
        out = model(x, modality=name)
        assert out["logits"].shape == (2, n_cls)


def test_swa_window_does_not_change_shape_for_long_seq():
    blk = SWABlock(hidden_dim=32, num_heads=4, window=16)
    x = torch.randn(2, 128, 32)
    y, _, _ = blk(x)
    assert y.shape == (2, 128, 32)
