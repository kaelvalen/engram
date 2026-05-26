"""Streaming-inference correctness: processing a sequence in one shot must equal
processing it in chunks with carried state. This exercises conv_state +
SSM/Delta state hand-off and is the property the old indexed-assignment scan
silently broke.

SWA layers are excluded here because the classification path never streams
attention (a true KV-cache equivalence test would be separate); SSD/S4D/Delta
are the genuinely recurrent mixers and must satisfy chunk == full.
"""

from __future__ import annotations

import pytest
import torch
from prism.config import ModalityConfig, PRISMConfig
from prism.model import PRISMBackbone


def _backbone(pattern, **kw):
    torch.manual_seed(0)
    cfg = PRISMConfig(
        hidden_dim=32,
        num_heads=4,
        num_layers=len(pattern),
        block_pattern=pattern,
        modalities=[ModalityConfig("x", 32, 2)],
        **kw,
    )
    return PRISMBackbone(cfg).eval()


@pytest.mark.parametrize(
    "pattern",
    [
        ["s4", "delta"],
        ["s4", "s4", "delta"],
        ["delta", "s4"],
    ],
)
@pytest.mark.parametrize("ssm_kind", ["ssd", "s4d_legacy"])
def test_chunked_equals_full(pattern, ssm_kind):
    bb = _backbone(pattern, ssm_kind=ssm_kind)
    x = torch.randn(2, 128, 32)

    y_full, _ = bb(x)

    split = 53  # deliberately not a chunk-size multiple
    y1, st = bb(x[:, :split])
    y2, _ = bb(x[:, split:], st)
    y_chunked = torch.cat([y1, y2], dim=1)

    torch.testing.assert_close(y_chunked, y_full, rtol=2e-3, atol=2e-3)


def test_long_sequence_two_halves():
    bb = _backbone(["s4", "delta", "s4", "delta"])
    x = torch.randn(1, 512, 32)
    y_full, _ = bb(x)
    y1, st = bb(x[:, :256])
    y2, _ = bb(x[:, 256:], st)
    torch.testing.assert_close(torch.cat([y1, y2], dim=1), y_full, rtol=3e-3, atol=3e-3)
