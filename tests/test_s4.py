from __future__ import annotations

import torch
from engram.modules.s4 import parallel_scan


def test_parallel_scan_equivalence():
    B, H, T, N = 2, 3, 15, 4  # T not a power of 2 to test padding
    a = torch.randn(B, H, T, N, dtype=torch.complex64)
    b = torch.randn(B, H, T, N, dtype=torch.complex64)

    # Base recurrent implementation check
    h_recurrent = torch.zeros_like(b)
    h_curr = torch.zeros(B, H, N, dtype=torch.complex64)
    for t in range(T):
        h_curr = a[:, :, t] * h_curr + b[:, :, t]
        h_recurrent[:, :, t] = h_curr

    # Parallel scan implementation
    h_parallel = parallel_scan(a.clone(), b.clone())

    torch.testing.assert_close(h_parallel, h_recurrent)
