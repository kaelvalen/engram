"""Tests for SGMS gathered execution (v2 conditional compute):
equivalence to masked dense execution under expert-local time semantics,
gradient propagation, and FLOP/token reduction.
"""

from __future__ import annotations

import torch
from sgms.block import SGMSBlock
from sgms.config import SGMSConfig


def _create_matched_blocks(**overrides):
    cfg_dense = SGMSConfig(
        hidden_dim=16,
        num_heads=2,
        num_layers=1,
        top_k=1,
        decay_on_skip=False,
        gdr_decay_on_skip=False,
        straight_through=True,
        execution_mode="dense_masked",
        **overrides,
    )
    cfg_gathered = SGMSConfig(
        hidden_dim=16,
        num_heads=2,
        num_layers=1,
        top_k=1,
        decay_on_skip=False,
        gdr_decay_on_skip=False,
        straight_through=True,
        execution_mode="gathered",
        **overrides,
    )
    torch.manual_seed(0)
    block_dense = SGMSBlock(cfg_dense, layer_idx=0).eval()
    block_gathered = SGMSBlock(cfg_gathered, layer_idx=0).eval()
    block_gathered.load_state_dict(block_dense.state_dict())
    return block_dense, block_gathered


def test_gathered_vs_dense_forward_equivalence():
    block_dense, block_gathered = _create_matched_blocks()
    torch.manual_seed(42)
    x = torch.randn(2, 12, 16)

    with torch.no_grad():
        y_dense, _, routing_dense = block_dense(x)
        y_gathered, _, routing_gathered = block_gathered(x)

    assert torch.equal(routing_dense.indices, routing_gathered.indices)
    max_diff = (y_dense - y_gathered).abs().max().item()
    assert max_diff < 1e-5, f"Gathered output differed from dense output: {max_diff}"


def test_gathered_backward_gradient_flow():
    _, block_gathered = _create_matched_blocks()
    block_gathered.train()
    x = torch.randn(2, 8, 16, requires_grad=True)

    y, _, _ = block_gathered(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None and x.grad.abs().sum() > 0
    assert block_gathered.router.weight.grad is not None
    assert block_gathered.router.weight.grad.abs().sum() > 0


def test_gathered_b_greater_than_1_streaming_state_handoff():
    """Regression test for B > 1 streaming state hand-off in gathered execution.

    Verifies:
    1. Carried expert states maintain batch dimension B across chunks.
    2. States are correctly sliced per batch item without overwriting earlier batch items.
    3. Processing sequence in chunks with carried state matches full one-shot sequence.
    """
    _, block_gathered = _create_matched_blocks()
    B, T, D = 4, 32, 16
    torch.manual_seed(123)
    x = torch.randn(B, T, D)

    # 1. Full one-shot execution
    with torch.no_grad():
        y_full, states_full, _ = block_gathered(x)

    # 2. Chunked streaming execution: chunk1 + chunk2
    split = 16
    with torch.no_grad():
        y1, states1, _ = block_gathered(x[:, :split], states=None)
        # Verify batch dimension for all state components after chunk 1
        for k, v in states1.items():
            if hasattr(v, "S"):  # DeltaState
                assert v.S.shape[0] == B, f"DeltaState batch dim is {v.S.shape[0]}, expected {B}"
            elif hasattr(v, "k"):  # SWAState
                assert v.k.shape[0] == B
            else:  # Tensor (SSD / Conv)
                assert v.shape[0] == B, f"Tensor state batch dim is {v.shape[0]}, expected {B}"

        y2, states2, _ = block_gathered(x[:, split:], states=states1)
        # Verify batch dimension for all state components after chunk 2
        for k, v in states2.items():
            if hasattr(v, "S"):
                assert v.S.shape[0] == B, f"DeltaState batch dim is {v.S.shape[0]}, expected {B}"
            elif hasattr(v, "k"):
                assert v.k.shape[0] == B
            else:
                assert v.shape[0] == B, f"Tensor state batch dim is {v.shape[0]}, expected {B}"

    y_chunked = torch.cat([y1, y2], dim=1)
    diff = (y_full - y_chunked).abs().max().item()
    assert diff < 1e-4, f"Chunked output differed from full output: {diff}"

    # Verify final states match full sequence final states
    for k in states_full:
        st_full = states_full[k]
        st_chunk = states2[k]
        if hasattr(st_full, "S"):
            torch.testing.assert_close(st_chunk.S, st_full.S, rtol=1e-4, atol=1e-4)
        elif hasattr(st_full, "k"):
            torch.testing.assert_close(st_chunk.k, st_full.k, rtol=1e-4, atol=1e-4)
        else:
            torch.testing.assert_close(st_chunk, st_full, rtol=1e-4, atol=1e-4)


def test_gathered_b_greater_than_1_vs_dense_chunked():
    """Verify chunked gathered execution matches chunked dense execution for B > 1."""
    block_dense, block_gathered = _create_matched_blocks()
    B, T, D = 3, 24, 16
    torch.manual_seed(42)
    x = torch.randn(B, T, D)

    split = 12
    with torch.no_grad():
        yd1, st_d1, _ = block_dense(x[:, :split], states=None)
        yd2, _, _ = block_dense(x[:, split:], states=st_d1)
        y_dense = torch.cat([yd1, yd2], dim=1)

        yg1, st_g1, _ = block_gathered(x[:, :split], states=None)
        yg2, _, _ = block_gathered(x[:, split:], states=st_g1)
        y_gathered = torch.cat([yg1, yg2], dim=1)

    diff = (y_dense - y_gathered).abs().max().item()
    assert diff < 1e-5, f"Gathered chunked differed from dense chunked: {diff}"

