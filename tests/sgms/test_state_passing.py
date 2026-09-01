"""Streaming state-passing (spec §3.6, §9): full-sequence forward ≡ chunked
forward with state hand-off, in float64.

Routing is per-token and causal, so streaming decisions must be identical
to full-sequence decisions; expert states carry across chunk boundaries
unchanged.
"""

from __future__ import annotations

import pytest
import torch
from sgms.config import SGMSConfig
from sgms.model import SGMSLM


def _model(dtype=torch.float64, seed=0, **overrides):
    cfg = SGMSConfig(
        hidden_dim=16,
        num_heads=2,
        num_layers=2,
        ssd_state_dim=8,
        delta_chunk_size=4,
        scan_backend="reference",
        **overrides,
    )
    torch.manual_seed(seed)
    return SGMSLM(cfg, vocab_size=32).to(dtype).eval()


@pytest.mark.parametrize("split", [53, 64, 97])
def test_chunked_equals_full(split):
    m = _model()
    ids = torch.randint(0, 32, (2, 128))
    full = m(ids)
    part1 = m(ids[:, :split])
    part2 = m(ids[:, split:], part1["states"])
    torch.testing.assert_close(
        torch.cat([part1["logits"], part2["logits"]], dim=1),
        full["logits"],
        rtol=1e-9,
        atol=1e-11,
    )


def test_three_way_split_uneven():
    m = _model()
    ids = torch.randint(0, 32, (1, 100))
    full = m(ids)
    st = None
    outs = []
    for lo, hi in [(0, 31), (31, 67), (67, 100)]:
        out = m(ids[:, lo:hi], st)
        outs.append(out["logits"])
        st = out["states"]
    torch.testing.assert_close(torch.cat(outs, dim=1), full["logits"], rtol=1e-9, atol=1e-11)


def test_token_by_token_decode_equals_full():
    """Full prefill ≡ T==1 decode steps with carried state (streaming)."""
    m = _model()
    ids = torch.randint(0, 32, (1, 24))
    full = m(ids)
    st = None
    outs = []
    for t in range(24):
        out = m(ids[:, t : t + 1], st)
        outs.append(out["logits"])
        st = out["states"]
    torch.testing.assert_close(torch.cat(outs, dim=1), full["logits"], rtol=1e-9, atol=1e-11)


def test_routing_decisions_identical_when_streaming():
    """Causal per-token routing ⇒ stream and full run pick the same experts."""
    m = _model()
    ids = torch.randint(0, 32, (2, 64))
    full = m(ids)
    part1 = m(ids[:, :37])
    part2 = m(ids[:, 37:], part1["states"])
    for layer in range(2):
        idx_full = full["routings"][layer].indices
        idx_stream = torch.cat(
            [part1["routings"][layer].indices, part2["routings"][layer].indices], dim=1
        )
        assert torch.equal(idx_full, idx_stream)


def test_shared_expert_state_passing():
    m = _model(shared_expert="ssd")
    ids = torch.randint(0, 32, (1, 64))
    full = m(ids)
    part1 = m(ids[:, :29])
    part2 = m(ids[:, 29:], part1["states"])
    torch.testing.assert_close(
        torch.cat([part1["logits"], part2["logits"]], dim=1),
        full["logits"],
        rtol=1e-9,
        atol=1e-11,
    )


def test_state_detach_breaks_graph():
    m = _model(dtype=torch.float32)
    ids = torch.randint(0, 32, (1, 16))
    out = m(ids)
    st = out["states"].detach()
    for v in st.values():
        s = v.S if hasattr(v, "S") else v
        assert not s.requires_grad
