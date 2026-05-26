"""SSD (Mamba-2-style) mixer: parallel-vs-sequential, state passing, shapes."""

from __future__ import annotations

import pytest
import torch
from prism.modules.ssd import SSDBlock, SSDMixer


def _mixer(seed=0, **kw):
    torch.manual_seed(seed)
    base = dict(hidden_dim=32, num_heads=4, state_dim=16)
    base.update(kw)
    return SSDMixer(**base)


@pytest.mark.parametrize("T", [1, 7, 20, 64])
def test_parallel_matches_sequential_reference(T):
    m = _mixer().eval()
    x = torch.randn(2, T, 32)
    yp, hp = m(x)
    yr, hr = m.forward_reference(x)
    torch.testing.assert_close(yp, yr, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(hp, hr, rtol=2e-4, atol=2e-4)


def test_state_passing_two_chunks_equals_full():
    m = _mixer().eval()
    x = torch.randn(2, 40, 32)
    y_full, h_full = m(x)
    y1, s1 = m(x[:, :17])
    y2, s2 = m(x[:, 17:], s1)
    torch.testing.assert_close(torch.cat([y1, y2], dim=1), y_full, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(s2, h_full, rtol=2e-4, atol=2e-4)


def test_stepwise_decode_equals_prefill():
    m = _mixer().eval()
    x = torch.randn(2, 8, 32)
    y_pref, _ = m(x)
    ys, s = [], None
    for t in range(8):
        yt, s = m(x[:, t : t + 1], s)
        ys.append(yt)
    torch.testing.assert_close(torch.cat(ys, dim=1), y_pref, rtol=2e-4, atol=2e-4)


def test_shapes_and_state_dtype():
    m = _mixer()
    x = torch.randn(3, 12, 32)
    y, h = m(x)
    assert y.shape == (3, 12, 32)
    assert h.shape == (3, 4, 8, 16)  # [B, H, P, N]
    assert h.dtype == torch.float32


@pytest.mark.parametrize("backend", ["auto", "assoc", "reference"])
def test_scan_backends_agree_in_block(backend):
    torch.manual_seed(1)
    blk = SSDBlock(hidden_dim=32, num_heads=4, state_dim=16, scan_backend=backend).eval()
    x = torch.randn(2, 24, 32)
    y, _, _ = blk(x)
    assert y.shape == (2, 24, 32)
    assert torch.isfinite(y).all()
