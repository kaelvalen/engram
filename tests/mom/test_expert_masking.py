"""Additive write-mask flags on the PRISM mixers (MoM spec §3.4).

The MoM router selects a memory primitive per token.  Under dense-masked
execution the routed expert must behave *as if it only saw its own
subsequence*:

SSD  — the write term is zeroed on a miss, so the update is
       ``s_t = a_t ⊙ s_{t-1} + 0``.  With ``decay_on_skip: true`` (spec D1,
       the default) the decay still applies on a miss; with ``false`` the
       state is frozen (``freeze_on_mask=True`` at the mixer level).
GDR  — ``k, v, β`` are masked, so the transition ``(I − β̃ k̃ k̃ᵀ)`` collapses
       to identity and the write vanishes.  The spec's GDR equation
       (Appendix A) has no α forget gate; to honour §3.4's "state passes
       through exactly" on PRISM's gated variant, α is neutralised to 1 on a
       miss when ``freeze_on_mask=True`` (MoM's default for GDR).

These tests pin the masked semantics against explicit per-token sequential
references in float64 (spec §9 tolerance: rtol=1e-10).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from prism.modules.delta import GatedDeltaRule
from prism.modules.ssd import SSDMixer

RTOL = 1e-10
ATOL = 1e-12


def _ssd(dtype: torch.dtype = torch.float64) -> SSDMixer:
    torch.manual_seed(0)
    return (
        SSDMixer(hidden_dim=16, num_heads=2, state_dim=8, scan_backend="reference").to(dtype).eval()
    )


def _gdr(dtype: torch.dtype = torch.float64) -> GatedDeltaRule:
    torch.manual_seed(0)
    return GatedDeltaRule(hidden_dim=16, num_heads=2, qk_norm=True, chunk_size=4).to(dtype).eval()


def _mask(B: int, T: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    g = torch.Generator().manual_seed(1234)
    return (torch.rand(B, T, generator=g) > 0.4).to(dtype)


def _ssd_masked_reference(mixer, x, mask, freeze_on_mask, state=None):
    """Per-token masked SSD recurrence built directly on the mixer's projections."""
    xv, Cc, a, dBx = mixer._project(x)  # a: [B,T,H], dBx: [B,T,H,P,N]
    B, T, H, P, N = dBx.shape
    h = state.to(x.dtype).clone() if state is not None else torch.zeros(B, H, P, N, dtype=x.dtype)
    outs = []
    for t in range(T):
        m_t = mask[:, t].view(B, 1, 1, 1)
        a_t = a[:, t].view(B, H, 1, 1)
        if freeze_on_mask:
            a_t = torch.where(m_t > 0, a_t, torch.ones_like(a_t))
        h = a_t * h + m_t * dBx[:, t]
        y_t = (h * Cc[:, t].unsqueeze(-2)).sum(-1) + mixer.D.view(1, H, 1) * xv[:, t]
        outs.append(y_t)
    y = torch.stack(outs, dim=1).reshape(B, T, -1).to(x.dtype)
    gate = F.silu(mixer.gate_proj(x))
    return mixer.out_proj(y * gate), h


def _gdr_masked_reference(mixer, x, mask, freeze_on_mask, state=None):
    """Per-token masked GDR recurrence built directly on the mixer's projections."""
    q, k, v, alpha, beta, gate = mixer._project(x)
    B, H, T, Dh = k.shape
    S = (
        state.S.to(x.dtype).clone()
        if state is not None
        else torch.zeros(B, H, Dh, Dh, dtype=x.dtype)
    )
    outs = []
    for t in range(T):
        m_t = mask[:, t].view(B, 1, 1)
        k_t = m_t * k[:, :, t]  # [B,H,Dh]
        v_t = m_t * v[:, :, t]
        b_t = (mask[:, t].view(B, 1) * beta[:, :, t]).view(B, H, 1, 1)
        a_t = alpha[:, :, t].unsqueeze(-1).unsqueeze(-1)
        if freeze_on_mask:
            a_t = torch.where(m_t.unsqueeze(-1) > 0, a_t, torch.ones_like(a_t))

        S_k = torch.einsum("bhij,bhj->bhi", S, k_t).unsqueeze(-1)
        outer_k = k_t.unsqueeze(-2)
        outer_v = v_t.unsqueeze(-1)
        S = a_t * (S - b_t * (S_k @ outer_k)) + b_t * (outer_v @ outer_k)
        outs.append(torch.einsum("bhij,bhj->bhi", S, q[:, :, t]))

    out = torch.stack(outs, dim=2).transpose(1, 2).contiguous().view(B, T, -1)
    return mixer.out_proj(out * gate), S


# ---------------------------------------------------------------------------
# SSD
# ---------------------------------------------------------------------------


def test_ssd_mask_all_ones_equals_unmasked():
    m = _ssd()
    x = torch.randn(2, 24, 16, dtype=torch.float64)
    ones = torch.ones(2, 24, dtype=torch.float64)
    y0, s0 = m(x)
    y1, s1 = m(x, write_mask=ones)
    torch.testing.assert_close(y1, y0, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(s1, s0, rtol=RTOL, atol=ATOL)


def test_ssd_masked_matches_sequential_reference():
    m = _ssd()
    x = torch.randn(2, 24, 16, dtype=torch.float64)
    mask = _mask(2, 24)
    y, s = m(x, write_mask=mask)  # decay_on_skip semantics (decay applies on miss)
    y_ref, s_ref = _ssd_masked_reference(m, x, mask, freeze_on_mask=False)
    torch.testing.assert_close(y, y_ref, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(s, s_ref, rtol=RTOL, atol=ATOL)


def test_ssd_masked_matches_reference_with_state_in():
    m = _ssd()
    x = torch.randn(2, 24, 16, dtype=torch.float64)
    mask = _mask(2, 24)
    s0 = torch.randn(2, 2, 8, 8, dtype=torch.float64) * 0.1
    y, s = m(x, s0, write_mask=mask)
    y_ref, s_ref = _ssd_masked_reference(m, x, mask, freeze_on_mask=False, state=s0)
    torch.testing.assert_close(y, y_ref, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(s, s_ref, rtol=RTOL, atol=ATOL)


def test_ssd_freeze_on_mask_keeps_state_on_full_miss():
    m = _ssd()
    x = torch.randn(2, 16, 16, dtype=torch.float64)
    s0 = torch.randn(2, 2, 8, 8, dtype=torch.float64) * 0.1
    zero = torch.zeros(2, 16, dtype=torch.float64)
    _, s = m(x, s0, write_mask=zero, freeze_on_mask=True)
    torch.testing.assert_close(s, s0, rtol=0.0, atol=0.0)


def test_ssd_decay_applies_on_full_miss_by_default():
    m = _ssd()
    x = torch.randn(1, 8, 16, dtype=torch.float64)
    s0 = torch.randn(1, 2, 8, 8, dtype=torch.float64) * 0.1
    zero = torch.zeros(1, 8, dtype=torch.float64)
    _, s = m(x, s0, write_mask=zero)  # freeze_on_mask=False → s_T = (Π a_t) ⊙ s_0
    _, _, a, _ = m._project(x)
    decay = a[0].prod(dim=0).view(1, 2, 1, 1)
    torch.testing.assert_close(s, decay * s0, rtol=RTOL, atol=ATOL)


def test_ssd_masked_decode_step_matches_reference():
    m = _ssd()
    x = torch.randn(2, 1, 16, dtype=torch.float64)
    mask = _mask(2, 1)
    s0 = torch.randn(2, 2, 8, 8, dtype=torch.float64) * 0.1
    y, s = m(x, s0, write_mask=mask, freeze_on_mask=True)
    y_ref, s_ref = _ssd_masked_reference(m, x, mask, freeze_on_mask=True, state=s0)
    torch.testing.assert_close(y, y_ref, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(s, s_ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# GDR
# ---------------------------------------------------------------------------


def test_gdr_mask_all_ones_equals_unmasked():
    m = _gdr()
    x = torch.randn(2, 20, 16, dtype=torch.float64)
    ones = torch.ones(2, 20, dtype=torch.float64)
    y0, s0 = m(x)
    y1, s1 = m(x, write_mask=ones)
    torch.testing.assert_close(y1, y0, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(s1.S, s0.S, rtol=RTOL, atol=ATOL)


def test_gdr_masked_matches_sequential_reference_frozen():
    m = _gdr()
    x = torch.randn(2, 20, 16, dtype=torch.float64)
    mask = _mask(2, 20)
    y, s = m(x, write_mask=mask, freeze_on_mask=True)
    y_ref, s_ref = _gdr_masked_reference(m, x, mask, freeze_on_mask=True)
    torch.testing.assert_close(y, y_ref, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(s.S, s_ref, rtol=RTOL, atol=ATOL)


def test_gdr_masked_matches_sequential_reference_decaying():
    m = _gdr()
    x = torch.randn(2, 20, 16, dtype=torch.float64)
    mask = _mask(2, 20)
    y, s = m(x, write_mask=mask, freeze_on_mask=False)
    y_ref, s_ref = _gdr_masked_reference(m, x, mask, freeze_on_mask=False)
    torch.testing.assert_close(y, y_ref, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(s.S, s_ref, rtol=RTOL, atol=ATOL)


def test_gdr_miss_passes_state_through_exactly_when_frozen():
    m = _gdr()
    x = torch.randn(2, 12, 16, dtype=torch.float64)
    s0 = torch.randn(2, 2, 8, 8, dtype=torch.float64) * 0.1
    zero = torch.zeros(2, 12, dtype=torch.float64)
    from prism.modules.delta import DeltaState

    _, s = m(x, DeltaState(S=s0), write_mask=zero, freeze_on_mask=True)
    torch.testing.assert_close(s.S, s0, rtol=0.0, atol=0.0)


def test_gdr_masked_decode_step_matches_reference():
    m = _gdr()
    x = torch.randn(2, 1, 16, dtype=torch.float64)
    mask = _mask(2, 1)
    s0 = torch.randn(2, 2, 8, 8, dtype=torch.float64) * 0.1
    from prism.modules.delta import DeltaState

    y, s = m(x, DeltaState(S=s0), write_mask=mask, freeze_on_mask=True)
    y_ref, s_ref = _gdr_masked_reference(m, x, mask, freeze_on_mask=True, state=DeltaState(S=s0))
    torch.testing.assert_close(y, y_ref, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(s.S, s_ref, rtol=RTOL, atol=ATOL)
