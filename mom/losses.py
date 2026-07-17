"""Stability objectives (spec §3.7).

    L     = L_task + λ_bal · L_bal + λ_z · L_z
    L_bal = K · Σ_e f_e · P_e   (Switch-style; f hard counts, P mean softmax)
    L_z   = mean_t ( logsumexp(z_t) )²

Both are averaged over the layers that carry the required quantities —
layers without logits (uniform/random router modes) contribute nothing.
"""

from __future__ import annotations

import torch

from .router import RoutingOutput


def load_balancing_loss(routings: list[RoutingOutput]) -> torch.Tensor:
    """Mean over layers of K · Σ_e f_e · P_e.

    f_e = fraction of tokens with e ∈ S_t (hard count, non-differentiable);
    P_e = mean softmax probability assigned to e (differentiable).
    """
    terms = []
    for r in routings:
        if r.probs is None:
            continue
        K = r.mask.shape[-1]
        f = r.mask.float().mean(dim=(0, 1))  # [K], no gradient by construction
        P = r.probs.mean(dim=(0, 1))  # [K]
        terms.append(K * (f * P).sum())
    if not terms:
        return torch.zeros(())
    return torch.stack(terms).mean()


def router_z_loss(routings: list[RoutingOutput]) -> torch.Tensor:
    """Mean over layers (then tokens) of logsumexp(z_t)²."""
    terms = []
    for r in routings:
        if r.logits is None:
            continue
        terms.append(torch.logsumexp(r.logits, dim=-1).pow(2).mean())
    if not terms:
        return torch.zeros(())
    return torch.stack(terms).mean()


def mom_auxiliary_loss(
    routings: list[RoutingOutput], lambda_bal: float, lambda_z: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """λ_bal · L_bal + λ_z · L_z with detached components for logging."""
    bal = load_balancing_loss(routings)
    z = router_z_loss(routings)
    total = lambda_bal * bal + lambda_z * z
    return total, {"bal": bal.detach(), "z": z.detach()}
