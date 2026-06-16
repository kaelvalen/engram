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


def _fla_runs_on_gpu() -> bool:
    """Probe whether FLA's Triton kernel actually executes on this torch/CUDA pair.

    FLA may import cleanly but fail at runtime due to Triton/platform/torch
    version mismatches (e.g. its internal CPU fallback references torch.cpu.device
    which does not exist on some PyTorch builds).  We only trust the backend after
    a tiny end-to-end call succeeds.
    """
    if _FLA is None or not torch.cuda.is_available():
        return False
    try:
        B, T, H, K, V = 1, 16, 2, 8, 8
        q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device="cuda")
        g = torch.randn(B, T, H, dtype=torch.bfloat16, device="cuda")
        beta = torch.rand(B, T, H, dtype=torch.bfloat16, device="cuda")
        _FLA(q, k, v, g, beta, scale=1.0, head_first=False)
        return True
    except Exception:
        return False


_HAS_FLA = _fla_runs_on_gpu()

skip_no_fla = pytest.mark.skipif(
    not _HAS_FLA,
    reason="flash-linear-attention Triton kernel must execute for FLA equivalence test",
)


@skip_no_fla
@pytest.mark.parametrize("dtype,atol", [(torch.bfloat16, 1e-2)])
def test_fla_matches_reference(dtype, atol):
    torch.manual_seed(0)
    device = "cuda"
    delta = GatedDeltaRule(hidden_dim=64, num_heads=4, chunk_size=64).to(device=device, dtype=dtype)
    x = torch.randn(2, 64, 64, device=device, dtype=dtype)
    q, k, v, alpha, beta, _ = delta._project(x)
    S0 = torch.zeros(2, 4, 16, 16, device=device, dtype=torch.float32)

    o_ref, S_ref = delta._recurrent_vectorized(q, k, v, alpha, beta, S0, delta.chunk_size)
    o_fla, S_fla = delta._forward_fla(q, k, v, alpha, beta, S0)

    torch.testing.assert_close(o_fla.float(), o_ref.float(), rtol=1e-2, atol=atol)
    torch.testing.assert_close(S_fla.float(), S_ref.float(), rtol=1e-2, atol=atol)


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
