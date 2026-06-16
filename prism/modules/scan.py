"""Linear-recurrence parallel scan backends.

All functions solve the first-order linear recurrence

    h_t = a_t * h_{t-1} + b_t          (inclusive, h_0 implicitly 0)

elementwise along the time axis ``dim=-2`` for tensors shaped ``[..., T, N]``.
The combination operator on (decay, input) pairs is associative:

    (A1, B1) ∘ (A2, B2) = (A2·A1, A2·B1 + B2)

Three backends are provided:

* :func:`seq_recurrence`        — sequential ground truth (Python loop).
* :func:`hillis_steele_recurrence` — fully vectorized, no indexed assignment.
* :func:`assoc_recurrence`      — :func:`torch.associative_scan` (fused kernel)
                                  with automatic fallback to Hillis-Steele.

The previous hand-written Blelloch up/down-sweep used strided indexed
assignment (``a[:, :, idx_r] = ...``), which compiles to non-contiguous
scatter writes that are 3–10× slower than reshape-based ops on modern GPUs.
The Hillis-Steele formulation here uses only ``F.pad`` + slicing, eliminating
that anti-pattern; the production path prefers ``torch.associative_scan``.
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


def seq_recurrence(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Sequential reference: h_t = a_t·h_{t-1} + b_t. Scan along dim=-2."""
    T = a.shape[-2]
    h_prev = torch.zeros_like(a[..., 0, :])
    out = []
    for t in range(T):
        h_prev = a[..., t, :] * h_prev + b[..., t, :]
        out.append(h_prev)
    return torch.stack(out, dim=-2)


def hillis_steele_recurrence(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vectorized inclusive scan via recursive doubling (Hillis-Steele).

    No indexed assignment: each level shifts by ``F.pad`` and combines with a
    pointwise op. O(T log T) work, fully parallel, correct for real or complex.
    """
    T = a.shape[-2]
    A = a
    B = b
    shift = 1
    # pad tuple is (last-dim-left, last-dim-right, time-left, time-right);
    # we only pad the time axis on the left by `shift`.
    while shift < T:
        # The leading `shift` rows must act as the identity element (decay=1,
        # input=0) so the combine is a no-op there. Built with cat so it works
        # for real and complex dtypes alike (F.pad's constant value is real-only).
        # A and B may have different trailing dims (e.g. decay is broadcast over
        # the state dimension), so pad each to its own width.
        a_lead_shape = (*A.shape[:-2], shift, A.shape[-1])
        b_lead_shape = (*B.shape[:-2], shift, B.shape[-1])
        ones = torch.ones(a_lead_shape, dtype=A.dtype, device=A.device)
        zeros = torch.zeros(b_lead_shape, dtype=B.dtype, device=B.device)
        A_prev = torch.cat([ones, A], dim=-2)[..., :T, :]
        B_prev = torch.cat([zeros, B], dim=-2)[..., :T, :]
        # combine(prev, cur):  A_new = A_cur·A_prev ,  B_new = A_cur·B_prev + B_cur
        B = A * B_prev + B
        A = A * A_prev
        shift *= 2
    return B


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
