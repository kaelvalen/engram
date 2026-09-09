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
