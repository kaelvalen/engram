"""Core spec §9 gate: dense-masked SGMSBlock ≡ sequential token-by-token
reference, float64, rtol=1e-10.

The reference (sgms.reference.sequential_block_reference) routes every token
individually through the experts' T==1 decode paths; the block under test
runs the dense-masked chunked/scan paths.  Agreement of the two is the
definition of correct masked-dense execution (§3.4).
"""

from __future__ import annotations

import pytest
import torch
from sgms.block import SGMSBlock
from sgms.config import SGMSConfig
from sgms.reference import sequential_block_reference
from sgms.state import SHARED_KEY, ExpertStateDict

RTOL = 1e-10
ATOL = 1e-12


def _block(dtype=torch.float64, seed=0, **overrides):
    cfg = SGMSConfig(
        hidden_dim=16,
        num_heads=2,
        num_layers=1,
        ssd_state_dim=8,
        delta_chunk_size=4,
        scan_backend="reference",
        **overrides,
    )
    torch.manual_seed(seed)
    return SGMSBlock(cfg, layer_idx=0).to(dtype).eval()


def _states(block, B, scale=0.1):
    """Non-zero fp64 carried states (streaming hand-off) for equivalence tests."""
    st = block.empty_states(B, torch.device("cpu"), torch.float64)
    out = ExpertStateDict()
    for k, v in st.items():
        if hasattr(v, "S"):  # DeltaState
            v.S.data = torch.randn(v.S.shape, dtype=torch.float64) * scale
            out[k] = v
        else:
            out[k] = torch.randn(v.shape, dtype=torch.float64) * scale
    return out


@pytest.mark.parametrize("seed", [0, 1])
def test_equivalence_default_k1(seed):
    """D1 defaults: k=1, decay_on_skip=True, gdr pass-through frozen."""
    b = _block(seed=seed)
    x = torch.randn(2, 33, 16, dtype=torch.float64)  # T not a chunk multiple
    y_dense, up_dense, _ = b(x)
    y_ref, up_ref, _ = sequential_block_reference(b, x)
    torch.testing.assert_close(y_dense, y_ref, rtol=RTOL, atol=ATOL)
    for k in up_ref:
        a, r = up_dense[k], up_ref[k]
        torch.testing.assert_close(
            a.S if hasattr(a, "S") else a, r.S if hasattr(r, "S") else r, rtol=RTOL, atol=ATOL
        )


@pytest.mark.parametrize("seed", [0, 1])
def test_equivalence_decay_on_skip_false(seed):
    """Registered ablation (§6.3): both experts freeze on a miss."""
    b = _block(seed=seed, decay_on_skip=False, gdr_decay_on_skip=False)
    x = torch.randn(2, 33, 16, dtype=torch.float64)
    y_dense, _, _ = b(x)
    y_ref, _, _ = sequential_block_reference(b, x)
    torch.testing.assert_close(y_dense, y_ref, rtol=RTOL, atol=ATOL)


def test_equivalence_gdr_decay_on_skip_true():
    """D1-style treatment extended to GDR's α gate."""
    b = _block(gdr_decay_on_skip=True)
    x = torch.randn(2, 33, 16, dtype=torch.float64)
    y_dense, _, _ = b(x)
    y_ref, _, _ = sequential_block_reference(b, x)
    torch.testing.assert_close(y_dense, y_ref, rtol=RTOL, atol=ATOL)


def test_equivalence_top2():
    b = _block(top_k=2)
    x = torch.randn(2, 33, 16, dtype=torch.float64)
    y_dense, _, _ = b(x)
    y_ref, _, _ = sequential_block_reference(b, x)
    torch.testing.assert_close(y_dense, y_ref, rtol=RTOL, atol=ATOL)


def test_equivalence_shared_expert():
    b = _block(shared_expert="ssd")
    x = torch.randn(2, 33, 16, dtype=torch.float64)
    y_dense, up_dense, _ = b(x)
    y_ref, up_ref, _ = sequential_block_reference(b, x)
    torch.testing.assert_close(y_dense, y_ref, rtol=RTOL, atol=ATOL)
    assert (0, SHARED_KEY) in up_dense and (0, SHARED_KEY) in up_ref


def test_equivalence_with_state_in():
    """Non-zero carried states (chunk hand-off) keep exact equivalence."""
    b = _block()
    x = torch.randn(2, 21, 16, dtype=torch.float64)
    st = _states(b, 2)
    y_dense, up_dense, _ = b(x, st)
    y_ref, up_ref, _ = sequential_block_reference(b, x, st)
    torch.testing.assert_close(y_dense, y_ref, rtol=RTOL, atol=ATOL)
    for k in up_ref:
        a, r = up_dense[k], up_ref[k]
        torch.testing.assert_close(
            a.S if hasattr(a, "S") else a, r.S if hasattr(r, "S") else r, rtol=RTOL, atol=ATOL
        )


def test_equivalence_knockout():
    """Forced expert exclusion (§7.3) is exact in both paths."""
    b = _block()
    x = torch.randn(2, 17, 16, dtype=torch.float64)
    y_dense, _, routing = b(x, exclude={"gdr"})
    y_ref, _, _ = sequential_block_reference(b, x, exclude={"gdr"})
    torch.testing.assert_close(y_dense, y_ref, rtol=RTOL, atol=ATOL)
    assert (routing.mask[..., 1] == 0).all()  # gdr never selected


def test_equivalence_single_token():
    b = _block()
    x = torch.randn(2, 1, 16, dtype=torch.float64)
    y_dense, _, _ = b(x)
    y_ref, _, _ = sequential_block_reference(b, x)
    torch.testing.assert_close(y_dense, y_ref, rtol=RTOL, atol=ATOL)
