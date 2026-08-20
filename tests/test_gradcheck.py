"""Analytic-vs-numeric gradient checks in float64.

Catches stale-buffer / detach / wrong-backward bugs that forward-only tests
miss. All inputs are tiny so the double-precision checks stay fast.
"""

from __future__ import annotations

import torch
from engram.modules.delta import GatedDeltaRule
from engram.modules.scan import hillis_steele_recurrence, seq_recurrence
from engram.modules.ssd import SSDMixer


def test_gradcheck_hillis_steele_scan():
    torch.manual_seed(0)
    a = torch.rand(1, 1, 5, 2, dtype=torch.float64, requires_grad=True) * 0.9
    b = torch.randn(1, 1, 5, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(hillis_steele_recurrence, (a, b), atol=1e-6)


def test_gradcheck_seq_recurrence():
    torch.manual_seed(0)
    a = torch.rand(1, 1, 4, 2, dtype=torch.float64, requires_grad=True) * 0.9
    b = torch.randn(1, 1, 4, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(seq_recurrence, (a, b), atol=1e-6)


def test_gradcheck_ssd_mixer_wrt_input():
    torch.manual_seed(0)
    m = SSDMixer(hidden_dim=8, num_heads=2, state_dim=4).double()
    x = torch.randn(1, 6, 8, dtype=torch.float64, requires_grad=True)

    def fn(inp):
        return m(inp)[0]

    assert torch.autograd.gradcheck(fn, (x,), atol=1e-5, rtol=1e-3)


def test_gradcheck_delta_rule_wrt_input():
    torch.manual_seed(0)
    d = GatedDeltaRule(hidden_dim=8, num_heads=2, chunk_size=4).double()
    x = torch.randn(1, 6, 8, dtype=torch.float64, requires_grad=True)

    def fn(inp):
        return d(inp)[0]

    assert torch.autograd.gradcheck(fn, (x,), atol=1e-5, rtol=1e-3)
