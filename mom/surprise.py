"""Lightweight causal standalone surprise predictor (spec §7.2, option (b)).

Layer-local: one ``HiddenSurprisePredictor`` lives inside each ``MoMBlock`` and
predicts that block's *own input* hidden stream ``x`` from its past —
``ĥ_t = P(x_{t-1})`` — producing a normalized per-token ``surprise`` fed only to
that block's router. Acyclic and single-pass: surprise uses only ``x``, which is
already causally produced before the router runs in the same forward.

No LatentEncoder; mirrors the EMA-baseline pattern of
``engram.saber.saber.Predictor`` (stable EMA shadow, online weights train toward
the target via an MSE ``pred_loss``) and the running-statistics normalization of
``engram.saber.saber.SurpriseEstimator``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SurprisePredictorConfig:
    hidden_dim: int
    predictor_hidden_dim: int = 64  # small MLP capacity
    ema_decay: float = 0.999  # stable-baseline lag
    surprise_mu_lambda: float = 0.99  # running-mean decay
    surprise_sigma_lambda: float = 0.99  # running-var decay
    surprise_eps_min: float = 1e-6
    surprise_eps_scale: float = 1.0
    surprise_max: float = 3.0


class HiddenSurprisePredictor(nn.Module):
    """ĥ_t = P(x_{t-1}); surprise = normalized |x_t − ĥ_t| (EMA baseline)."""

    def __init__(self, cfg: SurprisePredictorConfig):
        super().__init__()
        self.cfg = cfg
        # Online predictor, strictly past-only input.
        self.predictor = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.predictor_hidden_dim, cfg.hidden_dim),
        )
        # EMA shadows of all online predictor params: [w1, b1, w2, b2].
        # The stable surprise baseline; never updated by backprop.
        self._ema_shadows = [p.detach().clone() for p in self.predictor.parameters()]
        # Running surprise statistics, updated ONLY in training mode.
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("sigma", torch.ones(1))

    @torch.no_grad()
    def update_ema(self):
        """Blend online params into their EMA shadows (stable baseline)."""
        ema = self.cfg.ema_decay
        for shadow, p in zip(self._ema_shadows, self.predictor.parameters()):
            shadow.mul_(ema).add_(p.data, alpha=1 - ema)

    def _predict_with_ema(self, x_prev: torch.Tensor) -> torch.Tensor:
        """Forward with the EMA-shadowed weights: the stable, detached baseline."""
        w1, b1, w2, b2 = self._ema_shadows
        h = F.gelu(F.linear(x_prev, w1, b1))
        return F.linear(h, w2, b2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """x: [B, T, D] this block's own input hidden stream.

        Returns (surprise [B, T], aux with x_hat and pred_loss).
        """
        # Causal shift: pred_t uses x_{t-1}; t=0 predicts from zeros (<start>).
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)

        x_hat = self._predict_with_ema(x_prev)  # stable EMA baseline
        abs_diff = (x - x_hat).abs().mean(dim=-1)  # [B, T]

        if self.training:
            with torch.no_grad():
                self.mu.mul_(self.cfg.surprise_mu_lambda).add_(
                    abs_diff.mean(), alpha=1 - self.cfg.surprise_mu_lambda
                )
                centered = abs_diff - self.mu
                self.sigma.mul_(self.cfg.surprise_sigma_lambda).add_(
                    (centered**2).mean(), alpha=1 - self.cfg.surprise_sigma_lambda
                )

        eps = torch.clamp(
            self.cfg.surprise_eps_scale * self.sigma.sqrt(), min=self.cfg.surprise_eps_min
        )
        surprise = ((abs_diff - self.mu) / (self.sigma.sqrt() + eps)).clamp(
            0, self.cfg.surprise_max
        )

        # Online predictor learns toward the target (stop-grad target, JEPA-style).
        pred_online = self.predictor(x_prev)
        pred_loss = F.mse_loss(pred_online, x.detach())

        return surprise, {"x_hat": x_hat, "pred_loss": pred_loss}
