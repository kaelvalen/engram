"""Hand-derived Blelloch (1990) work-efficient parallel scan — preserved.

This is the original from-scratch up-sweep / down-sweep implementation. It is
kept as a *teaching and numerical-equivalence reference* only: the production
path (``prism.modules.scan``) now uses ``torch.associative_scan`` (or a
vectorized Hillis-Steele fallback), because the strided indexed assignment in
the sweeps below (``a[:, :, idx_r] = ...``) lowers to non-contiguous scatter
writes that are markedly slower than reshape-based ops on GPU.

Equivalence between this reference and the production scans is asserted in
``tests/test_scan_equivalence.py``.
"""

from __future__ import annotations

import math

import torch


def blelloch_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Parallel prefix scan: h_t = a_t * h_{t-1} + b_t.

    a, b: [B, H, T, N]  (complex or real). Returns h: [B, H, T, N].
    Combination operator: (A1,B1)⊕(A2,B2) = (A2*A1, A2*B1+B2).
    """
    B, H, T, N = a.shape

    a_orig = a.clone()
    b_orig = b.clone()

    T_pad = 1
    while T_pad < T:
        T_pad *= 2

    if T_pad > T:
        pad = T_pad - T
        a = torch.cat([a, torch.ones(B, H, pad, N, dtype=a.dtype, device=a.device)], dim=2)
        b = torch.cat([b, torch.zeros(B, H, pad, N, dtype=b.dtype, device=b.device)], dim=2)

    levels = int(math.log2(T_pad))

    # upsweep
    for d in range(levels):
        step = 2 ** (d + 1)
        idx_r = torch.arange(step - 1, T_pad, step, device=a.device)
        idx_l = idx_r - 2**d
        a_l = a[:, :, idx_l]
        b_l = b[:, :, idx_l]
        b[:, :, idx_r] = a[:, :, idx_r] * b_l + b[:, :, idx_r]
        a[:, :, idx_r] = a[:, :, idx_r] * a_l

    # downsweep (exclusive scan)
    a = a.clone()
    b = b.clone()
    a[:, :, -1] = torch.ones(B, H, N, dtype=a.dtype, device=a.device)
    b[:, :, -1] = torch.zeros(B, H, N, dtype=b.dtype, device=b.device)

    for d in range(levels - 1, -1, -1):
        step = 2 ** (d + 1)
        idx_r = torch.arange(step - 1, T_pad, step, device=a.device)
        idx_l = idx_r - 2**d
        a_l_old = a[:, :, idx_l].clone()
        b_l_old = b[:, :, idx_l].clone()

        a[:, :, idx_l] = a[:, :, idx_r]
        b[:, :, idx_l] = b[:, :, idx_r]

        b[:, :, idx_r] = a_l_old * b[:, :, idx_r] + b_l_old
        a[:, :, idx_r] = a_l_old * a[:, :, idx_r]

    # exclusive → inclusive
    return a_orig * b[:, :, :T] + b_orig
