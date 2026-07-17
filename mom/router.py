"""Per-token router (spec §3.3).

    z_t = W_r h_t (+ b_e)       W_r ∈ R^{K×D}, bias-free by default
    S_t = indices of the k largest z_t (default k=1, Switch-style)
    g_{t,e} = softmax(z_t)_e over e ∈ S_t, renormalised; 0 otherwise

Router parameters are per-layer and independent across layers.  Modes:
``learned`` (default), ``uniform`` (B4 — frozen g = 1/K ensemble) and
``random`` (B5 — seeded random per-token assignment, input-independent).

With k=1 the renormalised gate is exactly 1, so no gradient flows through
the gate path; the router then learns only via L_bal/L_z (§3.7), matching
Switch.  ``straight_through=True`` (R4 fallback) re-attaches the gate path
by treating the selected softmax probability as the gradient carrier.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .masking import topk_mask


@dataclass
class RoutingOutput:
    """Per-token routing decision for one layer.

    gates:   [B, T, K] renormalised gates, 0 outside each token's S_t
    mask:    [B, T, K] float {0,1} routing indicator m_{t,e}
    indices: [B, T, k] selected expert ids
    logits:  [B, T, K] raw z (None for non-learned modes)
    probs:   [B, T, K] full softmax over all experts (None for "random")
    """

    gates: torch.Tensor
    mask: torch.Tensor
    indices: torch.Tensor
    logits: torch.Tensor | None
    probs: torch.Tensor | None


class TokenRouter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 1,
        bias: bool = False,
        init_std: float = 0.01,
        straight_through: bool = False,
        mode: str = "learned",
        seed: int = 0,
    ):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
        if mode not in ("learned", "uniform", "random"):
            raise ValueError(f"unknown router mode: {mode!r}")
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.mode = mode
        self.straight_through = straight_through

        self.weight = nn.Parameter(torch.zeros(num_experts, hidden_dim))
        if mode == "learned":
            nn.init.normal_(self.weight, mean=0.0, std=init_std)
        else:
            self.weight.requires_grad_(False)  # frozen routers (B4/B5)

        self.router_bias = nn.Parameter(torch.zeros(num_experts)) if bias else None

        # Seeded generator for mode="random" (B5): fixed seed ⇒ reproducible
        # assignments across reruns, independent of input values.
        self._generator: torch.Generator | None = None
        if mode == "random":
            self._generator = torch.Generator()
            self._generator.manual_seed(seed)

    def forward(self, h: torch.Tensor, exclude: set[int] | None = None) -> RoutingOutput:
        """Route every token. h: [B, T, D] pre-norm hidden state (§3.3).

        ``exclude`` force-drops experts (analysis §7.3 knockout); the router
        renormalises over the remaining ones. Ignored outside "learned" mode.
        """
        B, T, _ = h.shape
        K, k = self.num_experts, self.top_k

        if self.mode == "uniform":
            idx = torch.arange(K, device=h.device).view(1, 1, K).expand(B, T, K)
            gates = torch.full((B, T, K), 1.0 / K, dtype=h.dtype, device=h.device)
            mask = torch.ones(B, T, K, dtype=torch.float32, device=h.device)
            return RoutingOutput(gates, mask, idx.clone(), None, gates.clone())

        if self.mode == "random":
            scores = torch.rand(B, T, K, generator=self._generator, device=h.device)
            idx = scores.topk(k, dim=-1).indices
            mask = topk_mask(idx, K).to(h.dtype)
            gates = mask / k
            return RoutingOutput(gates, mask, idx, None, None)

        z = F.linear(h, self.weight, self.router_bias)  # [B, T, K]
        if exclude:
            if len(exclude) > K - k:
                raise ValueError(f"exclude={sorted(exclude)} leaves < top_k={k} experts")
            drop = torch.zeros(K, dtype=torch.bool, device=z.device)
            drop[list(exclude)] = True
            z = z.masked_fill(drop, float("-inf"))

        probs = F.softmax(z, dim=-1)
        idx = z.topk(k, dim=-1).indices  # [B, T, k]
        sel = probs.gather(-1, idx)  # softmax mass on the selected experts
        renorm = sel / sel.sum(dim=-1, keepdim=True)
        if self.straight_through:
            # Forward value = hard renormalised gate; gradient as if the gate
            # were the (pre-renormalised) selected softmax probability.
            gates_sel = renorm.detach() + sel - sel.detach()
        else:
            gates_sel = renorm
        gates = torch.zeros_like(probs).scatter(-1, idx, gates_sel)
        mask = topk_mask(idx, K)
        return RoutingOutput(gates, mask, idx, z, probs)


def routing_stats(routings: list[RoutingOutput]) -> dict:
    """First-class routing metrics (spec §6.1), computed per layer.

    Returns per-layer expert utilization, routing entropy, gate confidence
    (mean max-prob) and the minimum utilization (collapse detector, §6.4).
    """
    layers = []
    for r in routings:
        util = r.mask.float().mean(dim=(0, 1))  # [K]
        if r.probs is not None:
            p = r.probs.clamp_min(1e-12)
            entropy = -(p * p.log()).sum(-1).mean()
            confidence = r.probs.max(-1).values.mean()
        else:
            entropy = torch.zeros(())
            confidence = torch.zeros(())
        layers.append(
            {
                "utilization": util.detach().cpu(),
                "entropy": float(entropy),
                "gate_confidence": float(confidence),
                "min_utilization": float(util.min()),
            }
        )
    min_util = min(layer["min_utilization"] for layer in layers) if layers else 0.0
    return {"layers": layers, "min_utilization": min_util}
