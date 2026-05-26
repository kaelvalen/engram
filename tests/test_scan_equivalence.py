"""Numerical equivalence of the linear-recurrence scan backends.

Anchors the production scans (Hillis-Steele, torch.associative_scan) and the
preserved hand-derived Blelloch reference against the sequential ground truth.
"""

from __future__ import annotations

import pytest
import torch
from prism.modules.scan import (
    assoc_recurrence,
    hillis_steele_recurrence,
    linear_recurrence,
    seq_recurrence,
)
from prism.modules.scan_reference import blelloch_scan


def _inputs(B=2, H=3, T=15, N=4, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    if dtype == torch.complex64:
        a = torch.randn(B, H, T, N, dtype=dtype) * 0.5
        b = torch.randn(B, H, T, N, dtype=dtype)
    else:
        a = torch.sigmoid(torch.randn(B, H, T, N))  # decay in (0,1) → stable
        b = torch.randn(B, H, T, N)
    return a, b


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("T", [1, 8, 15, 16, 31])
def test_hillis_steele_matches_sequential(dtype, T):
    a, b = _inputs(T=T, dtype=dtype)
    torch.testing.assert_close(hillis_steele_recurrence(a, b), seq_recurrence(a, b))


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("T", [8, 15, 16])
def test_assoc_matches_sequential(dtype, T):
    a, b = _inputs(T=T, dtype=dtype)
    torch.testing.assert_close(assoc_recurrence(a, b), seq_recurrence(a, b), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("T", [1, 7, 15, 16, 33])
def test_blelloch_reference_matches_sequential(dtype, T):
    """The preserved hand-derived Blelloch up/down-sweep is still correct."""
    a, b = _inputs(T=T, dtype=dtype)
    torch.testing.assert_close(blelloch_scan(a.clone(), b.clone()), seq_recurrence(a, b))


def test_dispatcher_backends_agree():
    a, b = _inputs(T=20)
    ref = seq_recurrence(a, b)
    for backend in ("auto", "assoc", "reference"):
        torch.testing.assert_close(
            linear_recurrence(a, b, backend), ref, rtol=1e-4, atol=1e-5
        )


def test_unknown_backend_raises():
    a, b = _inputs(T=4)
    with pytest.raises(ValueError, match="Unknown scan backend"):
        linear_recurrence(a, b, "nope")
