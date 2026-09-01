"""Lightweight standalone surprise predictor — per-layer, local (design
decision (b), EXPERIMENTS.md "wiring decision"). Unlike SABER's full
pipeline (LatentEncoder -> EMA Predictor -> SurpriseEstimator on encoded
z_t), this predicts each SGMSBlock's own raw pre-norm hidden stream x_t
directly from x_{t-1} -- single forward pass, no cross-layer dependency,
no cycle.

    x_hat_t = P_ema(x_{t-1})           # shift-by-one, causal by construction
    surprise_t = clamp((|x_t - x_hat_t|.mean(-1) - mu) / (sigma.sqrt()+eps), -max, max)

Same normalization scheme as SABER's SurpriseEstimator (running mu/sigma), but
**centered / signed**: SABER clamps to [0, max] (forcing nonneg), which when fed
through a per-expert router weight creates a constant baseline bias that
collapses routing at scale >> router logits. Here we keep the signed normalized
deviation (mean ~ 0 over time, negative for more-predictable-than-typical
tokens) so `w_e * surprise` perturbs routing per-token instead of biasing it
globally.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class SurprisePredictor(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        predictor_hidden_dim: int | None = None,
        ema_decay: float = 0.99,
        surprise_max: float = 5.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ema_decay = ema_decay
        self.surprise_max = surprise_max
        self.eps = eps
        ph = max(1, hidden_dim // 4) if predictor_hidden_dim is None else predictor_hidden_dim
        # Online predictor; its EMA copy is the stable surprise baseline.
        self.online = nn.Sequential(
            nn.Linear(hidden_dim, ph),
            nn.GELU(),
            nn.Linear(ph, hidden_dim),
        )
        self.ema = copy.deepcopy(self.online)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        # Running surprise statistics (SABER SurpriseEstimator scheme).
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("sigma", torch.ones(1))

    @torch.no_grad()
    def update_ema(self):
        """Standard param EMA step; call after optimizer.step()."""
        for e, o in zip(self.ema.parameters(), self.online.parameters()):
            e.mul_(self.ema_decay).add_(o.data, alpha=1 - self.ema_decay)

    def _shift(self, x: torch.Tensor) -> torch.Tensor:
        """x_prev_t = x_{t-1}; position 0 zero-padded (predicts from <start>)."""
        return torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]; returns centered, signed surprise [B, T] from the EMA
        (stable) baseline. Signed (mean ~ 0, can be negative) so a per-expert
        router weight modulates token-to-token *deviation* rather than adding a
        constant bias — fixes the routing-collapse failure at large scale.

        Running mu/sigma update only in training mode; the EMA baseline is
        detached, so surprise never backprops into the predictor here (it is a
        diagnostic feature, not a trainable head on this path).
        """
        x_prev = self._shift(x)
        x_hat = self.ema(x_prev)  # stable baseline, requires_grad=False
        abs_diff = (x - x_hat).abs().mean(dim=-1)  # [B, T]
        if self.training:
            with torch.no_grad():
                self.mu.mul_(0.99).add_(abs_diff.mean(), alpha=0.01)
                centered = abs_diff - self.mu
                self.sigma.mul_(0.99).add_((centered**2).mean(), alpha=0.01)
        # Signed / centered: mean ~ 0, symmetric clamp.
        surprise = ((abs_diff - self.mu) / (self.sigma.sqrt() + self.eps)).clamp(
            -self.surprise_max, self.surprise_max
        )
        return surprise

    def predict_online(self, x: torch.Tensor) -> torch.Tensor:
        """Online predictor output — separate path for the aux MSE training loss,
        kept apart from forward() so surprise is always read from the EMA copy.
        """
        return self.online(self._shift(x))
