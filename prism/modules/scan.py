"""Linear-recurrence parallel scan backends.

All functions solve the first-order linear recurrence

    h_t = a_t * h_{t-1} + b_t          (inclusive, h_0 implicitly 0)

elementwise along the time axis ``dim=-2`` for tensors shaped ``[..., T, N]``.
The combination operator on (decay, input) pairs is associative:

    (A1, B1) ∘ (A2, B2) = (A2·A1, A2·B1 + B2)

Three backends are provided:

* :func:`seq_recurrence`        — sequential reference (Python loop).
* :func:`hillis_steele_recurrence` — fully vectorized recursive doubling.
* :func:`assoc_recurrence`      — :func:`torch.associative_scan` (fused kernel)
                                  with automatic fallback to Hillis-Steele.

The previous hand-written Blelloch up/down-sweep used strided indexed
assignment (``a[:, :, idx_r] = ...``), which compiles to non-contiguous
scatter writes that are 3–10× slower than reshape-based ops on modern GPUs.
The Hillis-Steele formulation here uses only in-place slicing inside a custom
autograd Function, keeping the graph correct while avoiding the large
per-level ``torch.cat`` allocations that blow up memory on long sequences.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Cache whether torch exposes a usable associative_scan HOP.
_ASSOC_FN = None
_ASSOC_CHECKED = False


def _get_assoc_fn():
    global _ASSOC_FN, _ASSOC_CHECKED
    if not _ASSOC_CHECKED:
        _ASSOC_CHECKED = True
        try:  # location moved across versions; try the stable import.
            from torch._higher_order_ops.associative_scan import associative_scan

            _ASSOC_FN = associative_scan
        except Exception:  # pragma: no cover - depends on torch build
            _ASSOC_FN = getattr(torch, "associative_scan", None)
    return _ASSOC_FN


def _hillis_steele_inplace(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pure math Hillis-Steele scan; no autograd tracking."""
    T = a.shape[-2]
    A = a.clone()
    B = b.clone()
    shift = 1
    while shift < T:
        B[..., shift:, :] = A[..., shift:, :] * B[..., :-shift, :] + B[..., shift:, :]
        A[..., shift:, :] = A[..., shift:, :] * A[..., :-shift, :]
        shift *= 2
    return B


def seq_recurrence(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Sequential reference: h_t = a_t·h_{t-1} + b_t. Scan along dim=-2."""
    T = a.shape[-2]
    h_prev = torch.zeros_like(a[..., 0, :])
    out = []
    for t in range(T):
        h_prev = a[..., t, :] * h_prev + b[..., t, :]
        out.append(h_prev)
    return torch.stack(out, dim=-2)


class _HillisSteeleScan(torch.autograd.Function):
    """Custom autograd wrapper around an in-place Hillis-Steele scan.

    The in-place forward is memory-efficient; the backward is another in-place
    scan over the reversed sequence.

    Forward:  h_t = a_t h_{t-1} + b_t,   h_0 = 0.
    Backward: r_s = g_s + a_{s+1} r_{s+1},   r_{T+1} = 0.
              grad_a_s = r_s · h_{s-1}
              grad_b_s = r_s
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h = _hillis_steele_inplace(a, b)
        ctx.save_for_backward(a, h)
        return h

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, h = ctx.saved_tensors
        T = a.shape[-2]

        # Reverse-scan coefficients: coeff_rev[t] = a_rev[t-1] for t>=1,
        # coeff_rev[0] = 1 (identity, so the leading row is just go_rev[0]).
        a_rev = a.flip(-2)
        ones = torch.ones_like(a_rev[..., :1, :])
        coeff_rev = torch.cat([ones, a_rev[..., :-1, :]], dim=-2)
        go_rev = grad_output.flip(-2)

        r_rev = _hillis_steele_inplace(coeff_rev, go_rev)
        r = r_rev.flip(-2)

        # grad_a_s = r_s * h_{s-1}; prepend zero state for h_0.
        h_prev = torch.zeros_like(h[..., :1, :])
        if T > 1:
            h_prev = torch.cat([h_prev, h[..., :-1, :]], dim=-2)
        grad_a = r * h_prev

        return grad_a, r


def hillis_steele_recurrence(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vectorized inclusive scan via recursive doubling (Hillis-Steele).

    Implemented as a custom autograd Function so the in-place recursive-doubling
    updates do not break PyTorch's version tracking.
    """
    return _HillisSteeleScan.apply(a, b)


def assoc_recurrence(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """torch.associative_scan fused path; falls back to Hillis-Steele.

    associative_scan is a prototype higher-order op (PyTorch >= 2.5) and its
    docs warn of possible miscompiles, so we guard with try/except and a
    Hillis-Steele fallback that is always numerically correct.
    """
    fn = _get_assoc_fn()
    if fn is None:
        return hillis_steele_recurrence(a, b)

    scan_dim = a.dim() - 2  # time axis as a positive index

    def combine(left, right):
        a_l, b_l = left
        a_r, b_r = right
        return (a_r * a_l, a_r * b_l + b_r)

    try:
        _, h = fn(combine, (a, b), dim=scan_dim, combine_mode="generic")
        return h
    except Exception as exc:
        logger.warning(
            "torch.associative_scan failed (%s: %s); falling back to Hillis-Steele.",
            type(exc).__name__,
            exc,
        )
        return hillis_steele_recurrence(a, b)


def linear_recurrence(a: torch.Tensor, b: torch.Tensor, backend: str = "auto") -> torch.Tensor:
    """Dispatch to the requested scan backend.

    backend: "auto" (assoc if available else reference), "assoc", "reference".
    """
    if backend == "reference":
        return hillis_steele_recurrence(a, b)
    if backend in ("auto", "assoc"):
        return assoc_recurrence(a, b)
    raise ValueError(f"Unknown scan backend: {backend!r}")
