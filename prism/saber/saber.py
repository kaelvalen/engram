from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SABERConfig


@dataclass
class SABERState:
    policy_state: torch.Tensor
    predictor_state: torch.Tensor
    surprise_mu: torch.Tensor
    surprise_sigma: torch.Tensor
    step: int
    budget_history: list[float]
    infonce_loss_history: list[float]


class LatentEncoder(nn.Module):
    def __init__(self, cfg: SABERConfig, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = cfg.encoder_hidden_dim
        self.output_dim = cfg.policy_state_dim

        layers = []
        layers.append(nn.Linear(input_dim, self.hidden_dim))
        layers.append(nn.LayerNorm(self.hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(cfg.encoder_dropout))

        for _ in range(cfg.encoder_num_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(cfg.encoder_dropout))

        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyState(nn.Module):
    def __init__(self, cfg: SABERConfig):
        super().__init__()
        self.state_dim = cfg.policy_state_dim
        self.latent_dim = cfg.policy_state_dim

        self.gru = nn.GRU(
            input_size=self.latent_dim,
            hidden_size=self.state_dim,
            num_layers=cfg.policy_num_layers,
            batch_first=True,
            dropout=cfg.policy_dropout if cfg.policy_num_layers > 1 else 0.0,
        )

        self.projection_head = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.LayerNorm(self.state_dim),
            nn.GELU(),
            nn.Linear(self.state_dim, self.latent_dim),
        )

    def forward(self, s_prev: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        B = z_t.shape[0]
        z_t = z_t.unsqueeze(1)
        s_prev = s_prev.unsqueeze(0)
        s_t, _ = self.gru(z_t, s_prev)
        return s_t.squeeze(1)

    def get_projection(self, s_t: torch.Tensor) -> torch.Tensor:
        return self.projection_head(s_t)


class Predictor(nn.Module):
    def __init__(self, cfg: SABERConfig):
        super().__init__()
        self.ema_decay = cfg.predictor_ema_decay
        self.linear = nn.Linear(cfg.policy_state_dim, cfg.policy_state_dim, bias=False)

        self.register_buffer("ema_weight", self.linear.weight.data.clone())

    @torch.no_grad()
    def update_ema(self):
        self.ema_weight.mul_(self.ema_decay).add_(
            self.linear.weight.data, alpha=1 - self.ema_decay
        )

    def forward(self, s_t: torch.Tensor, use_ema: bool = True) -> torch.Tensor:
        weight = self.ema_weight if use_ema else self.linear.weight
        return F.linear(s_t, weight)


class SurpriseEstimator(nn.Module):
    def __init__(self, cfg: SABERConfig):
        super().__init__()
        self.mu_lambda = cfg.surprise_mu_lambda
        self.sigma_lambda = cfg.surprise_sigma_lambda
        self.eps_min = cfg.surprise_eps_min
        self.eps_scale = cfg.surprise_eps_scale
        self.surprise_max = cfg.surprise_max

        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("sigma", torch.ones(1))

    def forward(self, z_t: torch.Tensor, z_hat_t: torch.Tensor) -> torch.Tensor:
        diff = z_t - z_hat_t
        abs_diff = diff.abs().mean(dim=-1)

        with torch.no_grad():
            self.mu.mul_(self.mu_lambda).add_(abs_diff.mean(), alpha=1 - self.mu_lambda)
            centered = abs_diff - self.mu
            self.sigma.mul_(self.sigma_lambda).add_(
                (centered**2).mean(), alpha=1 - self.sigma_lambda
            )

        eps_t = torch.clamp(self.eps_scale * self.sigma.sqrt(), min=self.eps_min)
        surprise = (abs_diff - self.mu) / (self.sigma.sqrt() + eps_t)
        surprise = surprise.clamp(0, self.surprise_max)

        return surprise


class TopKWithSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor, k: int) -> torch.Tensor:
        topk_vals, topk_idx = scores.topk(k, dim=-1)
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, topk_idx, 1.0)
        ctx.save_for_backward(mask)
        return scores * mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        mask, = ctx.saved_tensors
        return grad_output, None


class SparseMemoryActivation(nn.Module):
    def __init__(self, cfg: SABERConfig):
        super().__init__()
        self.num_slots = cfg.num_memory_slots
        self.slot_dim = cfg.memory_slot_dim
        self.temperature = cfg.topk_temperature

        self.slot_embeddings = nn.Parameter(
            torch.randn(self.num_slots, self.slot_dim) * 0.02
        )
        self.query_proj = nn.Linear(cfg.policy_state_dim, self.slot_dim)

    def forward(self, x: torch.Tensor, budget: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        k = budget.round().long().clamp(min=1, max=self.num_slots)

        q = self.query_proj(x)
        scores = torch.matmul(q, self.slot_embeddings.t()) / self.temperature

        active_slots = []
        active_weights = []

        for b in range(B):
            for t in range(T):
                kt = k[b, t].item()
                if kt > 0:
                    topk_scores = TopKWithSTE.apply(scores[b, t], kt)
                    active_slots.append(self.slot_embeddings[topk_scores.bool()])
                    active_weights.append(topk_scores[topk_scores.bool()])

        return torch.stack(active_slots) if active_slots else torch.empty(0), \
               torch.stack(active_weights) if active_weights else torch.empty(0)


class AdaptiveBudget(nn.Module):
    def __init__(self, cfg: SABERConfig):
        super().__init__()
        self.floor = cfg.budget_floor
        self.alpha = cfg.budget_alpha
        self.max_budget = cfg.budget_max

    def forward(self, surprise: torch.Tensor) -> torch.Tensor:
        budget = self.floor + self.alpha * surprise.clamp(0, self.max_budget)
        return budget.clamp(self.floor, self.floor + self.alpha * self.max_budget)


class SABER(nn.Module):
    def __init__(self, cfg: SABERConfig, input_dim: int):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim

        self.encoder = LatentEncoder(cfg, input_dim)
        self.policy = PolicyState(cfg)
        self.predictor = Predictor(cfg)
        self.surprise = SurpriseEstimator(cfg)
        self.budget = AdaptiveBudget(cfg)
        self.memory = SparseMemoryActivation(cfg)

        self.register_buffer("step", torch.tensor(0))
        self.register_buffer("beta", torch.tensor(cfg.infonce_beta_start))

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[SABERState] = None,
    ) -> tuple[torch.Tensor, SABERState, dict]:
        B, T, _ = x.shape

        if state is None:
            policy_state = torch.zeros(B, self.cfg.policy_state_dim, device=x.device)
            predictor_state = torch.zeros(B, self.cfg.policy_state_dim, device=x.device)
            surprise_mu = torch.zeros(1, device=x.device)
            surprise_sigma = torch.ones(1, device=x.device)
            step = 0
            budget_history = []
            infonce_loss_history = []
        else:
            policy_state = state.policy_state
            predictor_state = state.predictor_state
            surprise_mu = state.surprise_mu
            surprise_sigma = state.surprise_sigma
            step = state.step
            budget_history = state.budget_history
            infonce_loss_history = state.infonce_loss_history

        z_t = self.encoder(x)
        s_t = self.policy(policy_state, z_t)

        z_hat_t = self.predictor(s_t.detach(), use_ema=True)
        surprise_t = self.surprise(z_t, z_hat_t)

        budget_t = self.budget(surprise_t)

        active_slots, active_weights = self.memory(x, budget_t)

        infonce_loss = self._compute_infonce(s_t, z_t)

        budget_history.append(budget_t.mean().item())
        infonce_loss_history.append(infonce_loss.item())

        if self.training:
            self.predictor.update_ema()
            self._anneal_beta(step)

        new_state = SABERState(
            policy_state=s_t.detach(),
            predictor_state=predictor_state,
            surprise_mu=self.surprise.mu.detach(),
            surprise_sigma=self.surprise.sigma.detach(),
            step=step + 1,
            budget_history=budget_history[-1000:],
            infonce_loss_history=infonce_loss_history[-1000:],
        )

        aux = {
            "z_t": z_t,
            "s_t": s_t,
            "z_hat_t": z_hat_t,
            "surprise": surprise_t,
            "budget": budget_t,
            "infonce_loss": infonce_loss,
            "active_slots": active_slots,
            "active_weights": active_weights,
        }

        return active_weights, new_state, aux

    def _compute_infonce(self, s_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        B = s_t.shape[0]
        proj_s = self.policy.get_projection(s_t)

        sim_pos = (proj_s * z_t).sum(dim=-1) / self.cfg.infonce_temperature

        neg_idx = torch.randint(0, B, (B, self.cfg.infonce_num_negatives), device=s_t.device)
        neg_z = z_t[neg_idx]
        sim_neg = torch.matmul(proj_s.unsqueeze(1), neg_z.transpose(-2, -1)).squeeze(1) / self.cfg.infonce_temperature

        logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)
        labels = torch.zeros(B, dtype=torch.long, device=s_t.device)

        return F.cross_entropy(logits, labels)

    def _anneal_beta(self, step: int):
        if step < self.cfg.infonce_beta_anneal_steps:
            progress = step / self.cfg.infonce_beta_anneal_steps
            self.beta.data = torch.tensor(
                self.cfg.infonce_beta_start * (1 - progress) + self.cfg.infonce_beta_end * progress
            )
        else:
            self.beta.data = torch.tensor(self.cfg.infonce_beta_end)

    def get_param_groups(self) -> list[dict]:
        fast_params = list(self.policy.parameters()) + \
                      list(self.predictor.parameters()) + \
                      list(self.surprise.parameters()) + \
                      list(self.budget.parameters()) + \
                      list(self.memory.parameters())

        slow_params = list(self.encoder.parameters())

        memory_params = list(self.memory.slot_embeddings)

        return [
            {"params": fast_params, "lr": self.cfg.lr_fast, "name": "fast"},
            {"params": slow_params, "lr": self.cfg.lr_slow, "name": "slow"},
            {"params": memory_params, "lr": self.cfg.lr_memory, "name": "memory"},
        ]


class SABERBackbone(nn.Module):
    def __init__(self, saber: SABER, backbone: nn.Module, write_gate_gamma: float = 1.0):
        super().__init__()
        self.saber = saber
        self.backbone = backbone
        self.write_gate_gamma = write_gate_gamma

    def forward(
        self,
        x: torch.Tensor,
        saber_state: Optional[SABERState] = None,
        backbone_states: Optional[list] = None,
    ) -> tuple[torch.Tensor, SABERState, list, dict]:
        B, T, _ = x.shape

        active_weights, new_saber_state, aux = self.saber(x, saber_state)

        write_strength = torch.sigmoid(self.write_gate_gamma * aux["surprise"])

        if self.training and hasattr(self.backbone, "forward_with_memory"):
            y, new_backbone_states = self.backbone.forward_with_memory(
                x, active_weights, write_strength, backbone_states
            )
        else:
            y, new_backbone_states = self.backbone(x, backbone_states)

        return y, new_saber_state, new_backbone_states, aux