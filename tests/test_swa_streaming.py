"""SWA streaming tests: KV-cache state must make chunked ≡ full-sequence
forward (fp64, rtol=1e-10), with correct absolute RoPE positions and exact
window truncation.
"""

from __future__ import annotations

import pytest
import torch
from engram.modules.attention import SlidingWindowAttention, SWABlock
from torch.autograd import gradcheck

RTOL = 1e-10
ATOL = 1e-12


def _attn(dtype=torch.float64, window=4, seed=0):
    torch.manual_seed(seed)
    return SlidingWindowAttention(hidden_dim=16, num_heads=2, window=window).to(dtype).eval()


def _block(dtype=torch.float64, window=4, seed=0):
    torch.manual_seed(seed)
    return SWABlock(hidden_dim=16, num_heads=2, window=window).to(dtype).eval()


# ---------------------------------------------------------------------------
# chunked ≡ full
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("split", [3, 4, 7, 13, 20])
def test_chunked_equals_full(split):
    b = _block()
    x = torch.randn(2, 24, 16, dtype=torch.float64)
    y_full, _, _ = b(x)
    y1, conv1, mix1 = b(x[:, :split])
    y2, _, _ = b(x[:, split:], conv1, mix1)
    torch.testing.assert_close(torch.cat([y1, y2], dim=1), y_full, rtol=RTOL, atol=ATOL)


def test_token_by_token_decode_equals_full():
    b = _block()
    x = torch.randn(1, 16, 16, dtype=torch.float64)
    y_full, _, _ = b(x)
    conv, mix, outs = None, None, []
    for t in range(16):
        y_t, conv, mix = b(x[:, t : t + 1], conv, mix)
        outs.append(y_t)
    torch.testing.assert_close(torch.cat(outs, dim=1), y_full, rtol=RTOL, atol=ATOL)


def test_split_exactly_at_window_boundary():
    b = _block(window=4)
    x = torch.randn(1, 12, 16, dtype=torch.float64)
    y_full, _, _ = b(x)
    y1, conv1, mix1 = b(x[:, :4])
    y2, _, _ = b(x[:, 4:], conv1, mix1)
    torch.testing.assert_close(torch.cat([y1, y2], dim=1), y_full, rtol=RTOL, atol=ATOL)


def test_state_carries_exactly_window_keys():
    a = _attn(window=4)
    x = torch.randn(1, 10, 16, dtype=torch.float64)
    _, st = a(x)
    assert st.k.shape[2] == 4 and st.v.shape[2] == 4
    assert st.pos == 10


def test_absolute_rope_positions_after_offset():
    """A chunk at pos>0 must use absolute RoPE phases, not restart at 0."""
    a = _attn(window=8)
    x = torch.randn(1, 10, 16, dtype=torch.float64)
    y_full, _ = a(x)
    _, st = a(x[:, :6])
    y_rest, _ = a(x[:, 6:], st)
    torch.testing.assert_close(
        torch.cat([a(x[:, :6])[0], y_rest], dim=1), y_full, rtol=RTOL, atol=ATOL
    )


# ---------------------------------------------------------------------------
# window truncation
# ---------------------------------------------------------------------------


def test_old_token_beyond_window_has_no_influence():
    a = _attn(window=4)
    x = torch.randn(1, 10, 16, dtype=torch.float64)
    x_pert = x.clone()
    x_pert[:, 0] += 10.0  # position 0 is ≥ window away from positions ≥ 4
    y0, _ = a(x)
    y1, _ = a(x_pert)
    assert not torch.allclose(y0[:, 3], y1[:, 3])  # still inside window at t=3
    torch.testing.assert_close(y0[:, 4:], y1[:, 4:], rtol=0.0, atol=0.0)


def test_no_state_path_unchanged():
    """state=None full-sequence behaviour is the regression anchor."""
    a = _attn()
    x = torch.randn(2, 12, 16, dtype=torch.float64)
    y, st = a(x, None)
    assert st.pos == 12
    assert y.shape == (2, 12, 16)


# ---------------------------------------------------------------------------
# gradients / detach
# ---------------------------------------------------------------------------


def test_gradcheck_block_with_state():
    b = _block()
    x1 = torch.randn(1, 5, 16, dtype=torch.float64)
    _, conv1, mix1 = b(x1)
    x2 = torch.randn(1, 5, 16, dtype=torch.float64, requires_grad=True)
    assert gradcheck(lambda t: b(t, conv1, mix1)[0], (x2,), raise_exception=True)


def test_gradcheck_block_no_state():
    b = _block()
    x = torch.randn(1, 6, 16, dtype=torch.float64, requires_grad=True)
    assert gradcheck(lambda t: b(t)[0], (x,), raise_exception=True)


def test_state_detach():
    b = _block(dtype=torch.float32)
    x = torch.randn(1, 6, 16)
    _, _, mix = b(x)
    st = mix.detach()
    assert not st.k.requires_grad and not st.v.requires_grad
