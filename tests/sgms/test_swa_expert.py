"""SWA expert for SGMS (spec §3.4 SWA bullet, v2 activation).

Masked semantics: non-routed tokens are excluded from the expert's window
and from output gathering — the window slides over the *routed subsequence*
(the expert's own time axis), RoPE positions are subsequence positions, and
the KV cache holds the last `window` routed keys/values only.
"""

from __future__ import annotations

import pytest
import torch
from sgms.block import SGMSBlock
from sgms.config import SGMSConfig
from sgms.reference import sequential_block_reference
from sgms.registry import build_expert

RTOL = 1e-10
ATOL = 1e-12


def _cfg(**overrides):
    base = dict(
        hidden_dim=16,
        num_heads=2,
        num_layers=1,
        ssd_state_dim=8,
        delta_chunk_size=4,
        scan_backend="reference",
        swa_window=4,
    )
    base.update(overrides)
    return SGMSConfig(**base)


def _swa(cfg=None, seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    return build_expert("swa", cfg or _cfg()).to(dtype).eval()


def _mask(B, T, dtype=torch.float64):
    g = torch.Generator().manual_seed(1234)
    return (torch.rand(B, T, generator=g) > 0.4).to(dtype)


def _swa_masked_reference(expert, x, mask, state=None):
    """Token-by-token masked SWA with a python-list KV cache (ground truth)."""
    B, T, _ = x.shape
    H, Dh, W = expert.num_heads, expert.head_dim, expert.window
    pos_prev = state.pos if state is not None else torch.zeros(B, dtype=torch.long)
    cache_k = (
        [state.k[b, :, -int(min(pos_prev[b], W)) :] for b in range(B)]
        if state is not None
        else [None] * B
    )
    cache_v = (
        [state.v[b, :, -int(min(pos_prev[b], W)) :] for b in range(B)]
        if state is not None
        else [None] * B
    )
    outs = []
    for t in range(T):
        x_t = x[:, t : t + 1]
        qb, kb, vb = expert.qkv(x_t).split(expert.hidden_dim, dim=-1)
        outs_b = []
        for b in range(B):
            q = qb[b].view(1, H, Dh).transpose(0, 1)
            k = kb[b].view(1, H, Dh).transpose(0, 1)
            v = vb[b].view(1, H, Dh).transpose(0, 1)
            # subsequence position along the expert's stream (same formula
            # as the masked path: cumsum-1, clamped at 0)
            rank = int(mask[b, : t + 1].sum()) - 1
            pos = max(int(pos_prev[b]) + rank, 0)
            from engram.modules.attention import _apply_rope, _build_rope_cache

            cos, sin = _build_rope_cache(int(pos) + 1, Dh, x.device)
            cos = cos[int(pos) : int(pos) + 1].to(q.dtype)  # [1, Dh]
            sin = sin[int(pos) : int(pos) + 1].to(q.dtype)
            # q/k are (H, 1, Dh); _apply_rope wants (B, H, T, Dh) + [T, Dh]
            q = _apply_rope(q.unsqueeze(0), cos, sin).squeeze(0)
            k = _apply_rope(k.unsqueeze(0), cos, sin).squeeze(0)
            if int(mask[b, t]):
                ck = k if cache_k[b] is None else torch.cat([cache_k[b], k], dim=1)
                cv = v if cache_v[b] is None else torch.cat([cache_v[b], v], dim=1)
                cache_k[b], cache_v[b] = ck[:, -W:], cv[:, -W:]
            if cache_k[b] is None:
                outs_b.append(torch.zeros(H * Dh, dtype=x.dtype))
                continue
            att = (q @ cache_k[b].transpose(-2, -1)) / (Dh**0.5)
            o = (att.softmax(-1) @ cache_v[b]).transpose(0, 1).reshape(1, -1)
            outs_b.append(o.squeeze(0))
        outs.append(expert.out_proj(torch.stack(outs_b)))
    new_pos = pos_prev + mask.long().sum(dim=1)
    k_out = torch.zeros(B, H, W, Dh, dtype=x.dtype)
    v_out = torch.zeros(B, H, W, Dh, dtype=x.dtype)
    from engram.modules.attention import SWAState

    for b in range(B):
        if cache_k[b] is not None:
            kk, vv = cache_k[b], cache_v[b]
            k_out[b, :, -kk.shape[1] :] = kk
            v_out[b, :, -vv.shape[1] :] = vv
    return torch.stack(outs, dim=1), SWAState(k=k_out, v=v_out, pos=new_pos)


# ---------------------------------------------------------------------------
# registry / config activation
# ---------------------------------------------------------------------------


def test_swa_expert_builds_from_registry():
    expert = _swa()
    assert expert.window == 4


def test_v1_swa_config_now_activates():
    from sgms.baselines import build_model

    cfg = _cfg(experts=("ssd", "gdr", "swa"), num_layers=2)
    model = build_model("sgms", cfg, vocab_size=32)
    out = model(torch.randint(0, 32, (1, 8)))
    assert out["logits"].shape == (1, 8, 32)


# ---------------------------------------------------------------------------
# masked semantics
# ---------------------------------------------------------------------------


def test_mask_all_ones_equals_unmasked():
    e = _swa()
    x = torch.randn(2, 12, 16, dtype=torch.float64)
    ones = torch.ones(2, 12, dtype=torch.float64)
    y0, _ = e(x)
    y1, _ = e(x, write_mask=ones)
    torch.testing.assert_close(y1, y0, rtol=RTOL, atol=ATOL)


def test_masked_matches_sequential_reference():
    e = _swa()
    x = torch.randn(2, 14, 16, dtype=torch.float64)
    mask = _mask(2, 14)
    y, st = e(x, write_mask=mask)
    y_ref, st_ref = _swa_masked_reference(e, x, mask)
    torch.testing.assert_close(y, y_ref, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(st.k, st_ref.k, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(st.v, st_ref.v, rtol=RTOL, atol=ATOL)
    assert torch.equal(st.pos, st_ref.pos)


def test_masked_state_passing_chunked_equals_full():
    e = _swa()
    x = torch.randn(2, 21, 16, dtype=torch.float64)
    mask = _mask(2, 21)
    y_full, _ = e(x, write_mask=mask)
    y1, st1 = e(x[:, :9], write_mask=mask[:, :9])
    y2, _ = e(x[:, 9:], st1, write_mask=mask[:, 9:])
    torch.testing.assert_close(torch.cat([y1, y2], dim=1), y_full, rtol=RTOL, atol=ATOL)


def test_masked_reference_with_state_in():
    e = _swa()
    x = torch.randn(2, 11, 16, dtype=torch.float64)
    mask = _mask(2, 11)
    _, st = e(x[:, :5], write_mask=mask[:, :5])
    y, st2 = e(x[:, 5:], st, write_mask=mask[:, 5:])
    y_ref, st2_ref = _swa_masked_reference(e, x[:, 5:], mask[:, 5:], state=st)
    torch.testing.assert_close(y, y_ref, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(st2.k, st2_ref.k, rtol=RTOL, atol=ATOL)


def test_unrouted_tokens_have_no_influence():
    e = _swa()
    x = torch.randn(1, 10, 16, dtype=torch.float64)
    mask = torch.tensor([[1, 0, 1, 0, 1, 1, 0, 1, 0, 1]], dtype=torch.float64)
    x_pert = x.clone()
    x_pert[:, 1] += 100.0  # unrouted
    x_pert[:, 3] -= 100.0  # unrouted
    y0, st0 = e(x, write_mask=mask)
    y1, st1 = e(x_pert, write_mask=mask)
    routed = mask.bool()
    torch.testing.assert_close(y0[:, routed[0]], y1[:, routed[0]], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(st0.k, st1.k, rtol=RTOL, atol=ATOL)


def test_window_slides_over_routed_subsequence():
    """A routed key falls out of the window after `window` newer routed keys,
    regardless of how many unrouted tokens intervene."""
    e = _swa()
    x = torch.randn(1, 30, 16, dtype=torch.float64)
    # routed at t=0, then 4 more routed at t=20..24 (window=4): the t=0 key
    # must be invisible to the last routed query.
    mask = torch.zeros(1, 30, dtype=torch.float64)
    mask[0, [0, 20, 21, 22, 23]] = 1.0
    x_pert = x.clone()
    x_pert[:, 0] += 50.0
    y0, _ = e(x, write_mask=mask)
    y1, _ = e(x_pert, write_mask=mask)
    assert not torch.allclose(y0[:, 20], y1[:, 20])  # still visible at 1st of the new run
    torch.testing.assert_close(y0[:, 23], y1[:, 23], rtol=RTOL, atol=ATOL)  # fell out


def test_gradcheck_swa_masked():
    e = _swa()
    x = torch.randn(1, 8, 16, dtype=torch.float64, requires_grad=True)
    mask = _mask(1, 8)
    from torch.autograd import gradcheck

    assert gradcheck(lambda t: e(t, write_mask=mask)[0], (x,), raise_exception=True)


# ---------------------------------------------------------------------------
# block-level integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("experts", [("ssd", "swa"), ("gdr", "swa")])
def test_sgms_block_with_swa_masked_equivalence(experts):
    cfg = _cfg(experts=experts)
    torch.manual_seed(0)
    block = SGMSBlock(cfg, 0).to(torch.float64).eval()
    x = torch.randn(2, 17, 16, dtype=torch.float64)
    y_dense, up_dense, _ = block(x)
    y_ref, up_ref, _ = sequential_block_reference(block, x)
    torch.testing.assert_close(y_dense, y_ref, rtol=RTOL, atol=ATOL)
    for key in up_ref:
        a, r = up_dense[key], up_ref[key]
        if hasattr(a, "S"):
            torch.testing.assert_close(a.S, r.S, rtol=RTOL, atol=ATOL)
        elif hasattr(a, "k"):
            torch.testing.assert_close(a.k, r.k, rtol=RTOL, atol=ATOL)
            torch.testing.assert_close(a.v, r.v, rtol=RTOL, atol=ATOL)
        else:
            torch.testing.assert_close(a, r, rtol=RTOL, atol=ATOL)


def test_sgms_block_with_swa_state_passing():
    cfg = _cfg(experts=("gdr", "swa"), num_layers=2)
    torch.manual_seed(0)
    from sgms.model import SGMSLM

    model = SGMSLM(cfg, vocab_size=32).to(torch.float64).eval()
    ids = torch.randint(0, 32, (1, 24))
    full = model(ids)
    p1 = model(ids[:, :10])
    p2 = model(ids[:, 10:], p1["states"])
    torch.testing.assert_close(
        torch.cat([p1["logits"], p2["logits"]], dim=1), full["logits"], rtol=1e-9, atol=1e-11
    )
