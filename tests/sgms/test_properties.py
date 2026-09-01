"""Property-based tests (hypothesis, spec §9):

- router outputs are valid simplex vectors (any input, any mode);
- masks are idempotent;
- permuting the expert-registry order leaves block outputs unchanged
  (up to the corresponding router-row permutation).
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from sgms.block import SGMSBlock
from sgms.config import SGMSConfig
from sgms.masking import topk_mask
from sgms.router import TokenRouter


def _cfg(**overrides):
    return SGMSConfig(
        hidden_dim=16,
        num_heads=2,
        num_layers=1,
        ssd_state_dim=8,
        delta_chunk_size=4,
        scan_backend="reference",
        **overrides,
    )


@settings(derandomize=True, max_examples=12, deadline=None)
@given(
    k=st.sampled_from([1, 2]),
    mode=st.sampled_from(["learned", "uniform", "random"]),
    seed=st.integers(min_value=0, max_value=100),
)
def test_gates_always_form_a_simplex(k, mode, seed):
    torch.manual_seed(seed)
    r = TokenRouter(16, 3, top_k=k, mode=mode, seed=seed)
    out = r(torch.randn(2, 11, 16))
    assert (out.gates >= 0).all()
    torch.testing.assert_close(out.gates.sum(-1), torch.ones(2, 11), rtol=1e-5, atol=1e-6)
    assert (out.mask.sum(-1) == (3 if mode == "uniform" else k)).all()


@settings(derandomize=True, max_examples=12, deadline=None)
@given(
    B=st.integers(1, 3),
    T=st.integers(1, 12),
    K=st.integers(2, 5),
    k=st.integers(1, 2),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_mask_idempotent_and_binary(B, T, K, k, seed):
    k = min(k, K)
    g = torch.Generator().manual_seed(seed)
    idx = torch.stack([torch.randperm(K, generator=g)[:k] for _ in range(B * T)], dim=0).view(
        B, T, k
    )
    m = topk_mask(idx, K)
    assert set(m.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(m * m, m)
    assert (m.sum(-1) == k).all()


def _copy_expert(dst, src):
    dst.load_state_dict(src.state_dict())


@settings(derandomize=True, max_examples=4, deadline=None)
@given(seed=st.integers(min_value=0, max_value=100))
def test_expert_order_permutation_invariance(seed):
    """Permuting the registry order (and router rows with it) is a no-op."""
    torch.manual_seed(seed)
    cfg_ab = _cfg(experts=("ssd", "gdr"))
    cfg_ba = _cfg(experts=("gdr", "ssd"))
    block_ab = SGMSBlock(cfg_ab, 0).to(torch.float64).eval()
    block_ba = SGMSBlock(cfg_ba, 0).to(torch.float64).eval()

    with torch.no_grad():
        # same conv/ffn/norm weights
        block_ba.norm1.load_state_dict(block_ab.norm1.state_dict())
        block_ba.norm2.load_state_dict(block_ab.norm2.state_dict())
        block_ba.conv.load_state_dict(block_ab.conv.state_dict())
        block_ba.ffn.load_state_dict(block_ab.ffn.state_dict())
        # experts copied to their matching slots
        _copy_expert(block_ba.experts["ssd"], block_ab.experts["ssd"])
        _copy_expert(block_ba.experts["gdr"], block_ab.experts["gdr"])
        # router rows permuted to match the new expert order
        block_ba.router.weight.copy_(
            torch.stack([block_ab.router.weight[1], block_ab.router.weight[0]])
        )

    x = torch.randn(2, 17, 16, dtype=torch.float64)
    y_ab = block_ab(x)[0]
    y_ba = block_ba(x)[0]
    torch.testing.assert_close(y_ab, y_ba, rtol=1e-10, atol=1e-12)

    # and routing assignments agree under the permutation
    ra, rb = block_ab(x)[2], block_ba(x)[2]
    assert torch.equal(ra.mask[..., 0], rb.mask[..., 1])
    assert torch.equal(ra.mask[..., 1], rb.mask[..., 0])


@settings(derandomize=True, max_examples=6, deadline=None)
@given(seed=st.integers(min_value=0, max_value=100))
def test_router_determinism_property(seed):
    """Same seed + same input ⇒ identical routing decisions (§5.2)."""
    h = torch.randn(2, 9, 16)
    torch.manual_seed(seed)
    r1 = TokenRouter(16, 3, top_k=2)
    torch.manual_seed(seed)
    r2 = TokenRouter(16, 3, top_k=2)
    assert torch.equal(r1(h).indices, r2(h).indices)


@settings(derandomize=True, max_examples=6, deadline=None)
@given(
    T=st.integers(2, 40),
    split_frac=st.floats(min_value=0.2, max_value=0.8),
    seed=st.integers(min_value=0, max_value=100),
)
def test_state_passing_is_exact_at_any_split(T, split_frac, seed):
    """Chunked ≡ full at an arbitrary split point (block level, fp64)."""
    torch.manual_seed(seed)
    block = SGMSBlock(_cfg(), 0).to(torch.float64).eval()
    x = torch.randn(1, T, 16, dtype=torch.float64)
    split = max(1, min(T - 1, int(T * split_frac)))
    y_full, st_full, _ = block(x)
    y1, st1, _ = block(x[:, :split])
    y2, _, _ = block(x[:, split:], st1)
    torch.testing.assert_close(torch.cat([y1, y2], dim=1), y_full, rtol=1e-9, atol=1e-11)
