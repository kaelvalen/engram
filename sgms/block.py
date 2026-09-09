"""SGMS block (spec §3): expert bank + per-token router + gated combination.

Anatomy follows ENGRAM exactly (§3.5): pre-norm → short causal conv →
[routed bank of memory-primitive mixers] → residual, then pre-norm →
SwiGLU FFN → residual.  The router reads the pre-norm hidden state h_t
(§3.3).  Every routed expert sees the *same* conv output but only its own
token subsequence, via write-side masking (§3.4).  Outputs combine as
y_t = Σ_{e ∈ S_t} g_{t,e} · y_{t,e}; the optional shared SSD expert
(§3.7) is always on and adds its output ungated.

Dense-masking vs Gathered execution semantics note (§3.4):
- In dense-masked mode, all K experts execute over the full sequence, with
  state decaying at every wall-clock step under default decay_on_skip=True.
- In gathered mode, tokens are dispatched conditionally so only the chosen
  experts compute on active tokens, operating under expert-local event-time
  (state frozen across skipped tokens). Numerical equivalence holds when
  decay_on_skip=False and gdr_decay_on_skip=False.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from engram.modules.conv import ShortCausalConv1d
from engram.modules.ffn import SwiGLU
from engram.modules.norm import RMSNorm

from .config import SGMSConfig
from .masking import combine_expert_outputs
from .registry import build_expert, expert_empty_state, expert_forward
from .router import RoutingOutput, TokenRouter
from .state import CONV_KEY, SHARED_KEY, ExpertStateDict, cat_expert_states, slice_expert_state
from .surprise import SurprisePredictor


class SGMSBlock(nn.Module):
    """One SGMS layer: a bank of heterogeneous memory primitives + router."""

    def __init__(self, cfg: SGMSConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.norm1 = RMSNorm(cfg.hidden_dim)
        self.conv = ShortCausalConv1d(cfg.hidden_dim, cfg.conv_kernel_size)
        self.router = TokenRouter(
            hidden_dim=cfg.hidden_dim,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            bias=cfg.router_bias,
            init_std=cfg.router_init_std,
            straight_through=cfg.straight_through,
            mode=cfg.router_mode,
            seed=cfg.router_seed + 1000 * layer_idx,
            surprise_scale=cfg.router_surprise_scale,
            surprise_weight_init=cfg.router_surprise_weight,
            freeze_surprise_weight=cfg.freeze_surprise_weight,
        )
        self.experts = nn.ModuleDict({name: build_expert(name, cfg) for name in cfg.experts})
        self.shared = build_expert(cfg.shared_expert, cfg) if cfg.shared_expert else None
        # Layer-local surprise predictor (design (b)): predicts this block's own
        # input; off by default (cfg.use_surprise_predictor).
        self.surprise_predictor = (
            SurprisePredictor(cfg.hidden_dim) if cfg.use_surprise_predictor else None
        )
        self.norm2 = RMSNorm(cfg.hidden_dim)
        self.ffn = SwiGLU(cfg.hidden_dim, cfg.ffn_expand)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
        self.ffn_dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        states: ExpertStateDict | None = None,
        exclude: set[str] | None = None,
        surprise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ExpertStateDict, RoutingOutput]:
        """x: [B, T, D]. Returns (y, state_updates, routing).

        ``exclude`` knocks out experts by name (analysis §7.3); the router
        renormalises over the survivors. ``surprise`` (optional, [B, T]) is
        passed to the router when ``router_surprise_scale > 0``; None keeps
        the plain router behaviour.
        """
        i = self.layer_idx
        names = list(self.experts.keys())
        drop_idx: set[int] | None = None
        if exclude:
            unknown = exclude - set(names)
            if unknown:
                raise ValueError(f"cannot exclude unknown experts {sorted(unknown)}")
            drop_idx = {names.index(n) for n in exclude}

        r = x
        x_n = self.norm1(x)
        # Surprise source: an explicit external `surprise` overrides; otherwise the
        # layer-local predictor (if enabled) generates it from this block's input.
        if surprise is None and self.surprise_predictor is not None:
            surprise = self.surprise_predictor(x_n)
        routing = self.router(
            x_n, exclude=drop_idx, surprise=surprise
        )

        # Post-hoc renormalisation for non-learned modes (router-level
        # exclusion applies to the learned top-k only).
        if drop_idx is not None and self.cfg.router_mode != "learned":
            keep = torch.ones_like(routing.mask)
            keep[..., list(drop_idx)] = 0.0
            mask = routing.mask * keep
            gates = routing.gates * keep
            gates = gates / gates.sum(-1, keepdim=True).clamp_min(1e-12)
            routing = RoutingOutput(gates, mask, routing.indices, routing.logits, routing.probs)

        conv_in = states.get((i, CONV_KEY)) if states is not None else None
        x_c, conv_new = self.conv(x_n, conv_in)

        updates = ExpertStateDict({(i, CONV_KEY): conv_new})
        if self.cfg.execution_mode == "gathered":
            B, _, _ = x_c.shape
            y = torch.zeros_like(x_c)
            for e, (name, expert) in enumerate(self.experts.items()):
                st_in = states.get((i, name)) if states is not None else None
                st_out_list = []
                for b in range(B):
                    st_in_b = slice_expert_state(st_in, b)
                    m_b = routing.mask[b, :, e] > 0.5
                    active_idx = torch.where(m_b)[0]
                    if active_idx.numel() > 0:
                        x_sub = x_c[b : b + 1, active_idx]
                        y_sub, st_out_b = expert_forward(
                            name, expert, x_sub, st_in_b, None, self.cfg
                        )
                        gate_sub = routing.gates[b : b + 1, active_idx, e : e + 1]
                        y[b, active_idx] += (y_sub * gate_sub).squeeze(0)
                        st_out_list.append(st_out_b)
                    else:
                        if st_in_b is not None:
                            st_out_list.append(st_in_b)
                        else:
                            st_out_list.append(
                                expert_empty_state(name, expert, 1, x_c.device, x_c.dtype)
                            )
                updates[(i, name)] = cat_expert_states(st_out_list)
        else:
            outs = []
            for e, (name, expert) in enumerate(self.experts.items()):
                st_in = states.get((i, name)) if states is not None else None
                y_e, st_out = expert_forward(
                    name, expert, x_c, st_in, routing.mask[..., e], self.cfg
                )
                updates[(i, name)] = st_out
                outs.append(y_e)
            y = combine_expert_outputs(outs, routing.gates)

        if self.shared is not None:
            st_in = states.get((i, SHARED_KEY)) if states is not None else None
            y_s, st_s = expert_forward(
                self.cfg.shared_expert, self.shared, x_c, st_in, None, self.cfg
            )
            updates[(i, SHARED_KEY)] = st_s
            y = y + y_s  # always on, ungated (§3.7)

        x = r + self.dropout(y)
        x = x + self.ffn_dropout(self.ffn(self.norm2(x)))
        return x, updates, routing

    def empty_states(self, batch_size: int, device, dtype) -> ExpertStateDict:
        """Zero-initialised states for this layer (spec §3.6)."""
        i = self.layer_idx
        out = ExpertStateDict({(i, CONV_KEY): self.conv.empty_state(batch_size, device, dtype)})
        for name, expert in self.experts.items():
            out[(i, name)] = expert_empty_state(name, expert, batch_size, device, dtype)
        if self.shared is not None:
            out[(i, SHARED_KEY)] = expert_empty_state(
                self.cfg.shared_expert, self.shared, batch_size, device, dtype
            )
        return out
