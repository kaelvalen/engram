from __future__ import annotations

import torch
from prism.modules.delta import GatedDeltaRule


def _make_inputs(B=2, H=3, T=20, Dh=8, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, H, T, Dh)
    k = torch.randn(B, H, T, Dh)
    v = torch.randn(B, H, T, Dh)
    # alpha, beta in (0, 1) like sigmoid output
    alpha = torch.sigmoid(torch.randn(B, H, T) + 4.0)  # bias toward ~1
    beta = torch.sigmoid(torch.randn(B, H, T))
    S0 = torch.randn(B, H, Dh, Dh) * 0.1
    return q, k, v, alpha, beta, S0


def test_delta_vectorized_matches_naive_module_forward_projections():
    """Same projections as real forward; naive loop vs chunked path must agree."""
    torch.manual_seed(0)
    delta = GatedDeltaRule(hidden_dim=64, num_heads=4)
    x = torch.randn(2, 32, 64)
    q, k, v, alpha, beta, _ = delta._project(x)
    B = x.shape[0]
    Dh = delta.head_dim
    S0 = torch.zeros(B, delta.num_heads, Dh, Dh, dtype=torch.float32)

    out_old, _ = GatedDeltaRule._recurrent_naive(q, k, v, alpha, beta, S0)
    out_new, _ = GatedDeltaRule._recurrent_vectorized(q, k, v, alpha, beta, S0, delta.chunk_size)
    torch.testing.assert_close(out_new, out_old, rtol=2e-3, atol=1e-4)


def test_chunkwise_matches_recurrent():
    q, k, v, alpha, beta, S0 = _make_inputs(T=20, Dh=8)

    o_ref, S_ref = GatedDeltaRule._recurrent_naive(q, k, v, alpha, beta, S0)
    o_par, S_par = GatedDeltaRule._recurrent_vectorized(q, k, v, alpha, beta, S0, chunk_size=8)

    torch.testing.assert_close(o_par, o_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(S_par, S_ref, rtol=2e-3, atol=2e-3)


def test_chunkwise_matches_recurrent_T_not_divisible():
    # T=23 with chunk_size=8 → chunks of 8, 8, 7
    q, k, v, alpha, beta, S0 = _make_inputs(T=23, Dh=8, seed=1)

    o_ref, S_ref = GatedDeltaRule._recurrent_naive(q, k, v, alpha, beta, S0)
    o_par, S_par = GatedDeltaRule._recurrent_vectorized(q, k, v, alpha, beta, S0, chunk_size=8)

    torch.testing.assert_close(o_par, o_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(S_par, S_ref, rtol=2e-3, atol=2e-3)


def test_step_one_matches_recurrent():
    q, k, v, alpha, beta, S0 = _make_inputs(T=1, Dh=8, seed=2)

    o_ref, S_ref = GatedDeltaRule._recurrent_naive(q, k, v, alpha, beta, S0)
    o_one, S_one = GatedDeltaRule._step_one(q, k, v, alpha, beta, S0)

    torch.testing.assert_close(o_one, o_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(S_one, S_ref, rtol=1e-5, atol=1e-5)


def test_chunkwise_single_chunk():
    # T == chunk_size: one full chunk
    q, k, v, alpha, beta, S0 = _make_inputs(T=8, Dh=8, seed=3)

    o_ref, S_ref = GatedDeltaRule._recurrent_naive(q, k, v, alpha, beta, S0)
    o_par, S_par = GatedDeltaRule._recurrent_vectorized(q, k, v, alpha, beta, S0, chunk_size=8)

    torch.testing.assert_close(o_par, o_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(S_par, S_ref, rtol=2e-3, atol=2e-3)
