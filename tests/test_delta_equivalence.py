"""Delta-rule backend equivalence: FLA Triton kernel vs the from-scratch
reference chunked solve.

This is the test that proves the hand-derived UT-transform / triangular-solve
implementation is numerically equivalent to the published FLA kernel. It is
skipped automatically when FLA / CUDA are unavailable (CPU-only CI), but it is
the gate that must pass on GPU before trusting `delta_backend="fla"`.
"""

from __future__ import annotations

import pytest
import torch
from prism.modules.delta import GatedDeltaRule, _load_fla

_FLA = _load_fla()
_HAS_FLA = _FLA is not None and torch.cuda.is_available()

skip_no_fla = pytest.mark.skipif(
    not _HAS_FLA, reason="flash-linear-attention + CUDA required for FLA equivalence test"
)


@skip_no_fla
@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-4), (torch.bfloat16, 1e-2)])
def test_fla_matches_reference(dtype, atol):
    torch.manual_seed(0)
    device = "cuda"
    delta = GatedDeltaRule(hidden_dim=64, num_heads=4, chunk_size=64).to(device)
    x = torch.randn(2, 64, 64, device=device, dtype=dtype)
    q, k, v, alpha, beta, _ = delta._project(x)
    S0 = torch.zeros(2, 4, 16, 16, device=device)

    o_ref, S_ref = delta._recurrent_vectorized(q, k, v, alpha, beta, S0, delta.chunk_size)
    o_fla, S_fla = delta._forward_fla(q, k, v, alpha, beta, S0)

    rtol = 1e-3 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(o_fla.float(), o_ref.float(), rtol=rtol, atol=atol)
    torch.testing.assert_close(S_fla.float(), S_ref.float(), rtol=rtol, atol=atol)


def test_fla_backend_falls_back_gracefully_on_cpu():
    """With backend='fla' but no CUDA/FLA, forward must not crash (falls back)."""
    torch.manual_seed(0)
    delta = GatedDeltaRule(hidden_dim=32, num_heads=4, backend="fla")
    x = torch.randn(2, 20, 32)
    o, state = delta(x)
    assert o.shape == (2, 20, 32)
    assert torch.isfinite(o).all()

    # Equivalence of the fallback path with the explicit reference backend.
    delta_ref = GatedDeltaRule(hidden_dim=32, num_heads=4, backend="reference")
    delta_ref.load_state_dict(delta.state_dict())
    o_ref, _ = delta_ref(x)
    torch.testing.assert_close(o, o_ref, rtol=2e-3, atol=2e-3)
