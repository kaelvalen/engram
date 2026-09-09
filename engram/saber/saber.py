from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SABERConfig


@dataclass
class SABERState:
    """Streaming state for SABER: the GRU policy hidden plus bounded
    diagnostic histories (budget / InfoNCE running means)."""

    policy_hidden: torch.Tensor  # [num_layers, B, policy_state_dim]
    # Keep scalar history on the active device.  Converting each value to a
    # Python float in forward would synchronize CUDA once per batch.
    budget_history: list[torch.Tensor] = field(default_factory=list)
    infonce_loss_history: list[torch.Tensor] = field(default_factory=list)


class LatentEncoder(nn.Module):
    def __init__(self, cfg: SABERConfig, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = cfg.encoder_hidden_dim
        self.output_dim = cfg.policy_state_dim

        layers: list[nn.Module] = [
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.encoder_dropout),
        ]
        for _ in range(cfg.encoder_num_layers - 1):
            layers += [
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.encoder_dropout),
            ]
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyState(nn.Module):
    """Capacity-limited GRU policy state (the information bottleneck)."""

    def __init__(self, cfg: SABERConfig):
        super().__init__()
        self.state_dim = cfg.policy_state_dim
        self.num_layers = cfg.policy_num_layers

        self.gru = nn.GRU(
            input_size=self.state_dim,
            hidden_size=self.state_dim,
            num_layers=cfg.policy_num_layers,
            batch_first=True,
            dropout=cfg.policy_dropout if cfg.policy_num_layers > 1 else 0.0,
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.LayerNorm(self.state_dim),
            nn.GELU(),
            nn.Linear(self.state_dim, self.state_dim),
        )

    def forward(
        self, hidden: torch.Tensor | None, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """z: [B, T, D] latents; hidden: [num_layers, B, H] or None (zeros).

        Returns (s, new_hidden) with s: [B, T, H]."""
        return self.gru(z, hidden)

    def get_projection(self, s_t: torch.Tensor) -> torch.Tensor:
        return self.projection_head(s_t)


class Predictor(nn.Module):
    """JEPA-style latent predictor: an online weight trained toward the
    encoder latents, plus an EMA copy used as the stable surprise baseline."""

    def __init__(self, cfg: SABERConfig):
        super().__init__()
        self.ema_decay = cfg.predictor_ema_decay
        self.linear = nn.Linear(cfg.policy_state_dim, cfg.policy_state_dim, bias=False)
        self.register_buffer("ema_weight", self.linear.weight.data.clone())

    @torch.no_grad()
    def update_ema(self):
        self.ema_weight.mul_(self.ema_decay).add_(self.linear.weight.data, alpha=1 - self.ema_decay)

    def forward(self, s_t: torch.Tensor, use_ema: bool = True) -> torch.Tensor:
        weight = self.ema_weight if use_ema else self.linear.weight
        return F.linear(s_t, weight)


class SurpriseEstimator(nn.Module):
    """Normalized absolute prediction error, tracked with running μ/σ.

    Buffers update only in training mode - evaluation forwards must not
    contaminate the estimator's statistics.
    """

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
        abs_diff = (z_t - z_hat_t).abs().mean(dim=-1)  # [B, T]

        if self.training:
            with torch.no_grad():
                self.mu.mul_(self.mu_lambda).add_(abs_diff.mean(), alpha=1 - self.mu_lambda)
                centered = abs_diff - self.mu
                self.sigma.mul_(self.sigma_lambda).add_(
                    (centered**2).mean(), alpha=1 - self.sigma_lambda
                )

        eps_t = torch.clamp(self.eps_scale * self.sigma.sqrt(), min=self.eps_min)
        surprise = (abs_diff - self.mu) / (self.sigma.sqrt() + eps_t)
        return surprise.clamp(0, self.surprise_max)


class SparseMemoryActivation(nn.Module):
    """Vectorized top-k slot readout. Per token, the ``budget`` (≥1, ≤ M)
    highest-scoring slots are mixed with softmax weights that sum to 1 over
    the selected set - differentiable everywhere except through the integer
    counts themselves.
    """

    def __init__(self, cfg: SABERConfig):
        super().__init__()
        self.num_slots = cfg.num_memory_slots
        self.slot_dim = cfg.memory_slot_dim
        self.temperature = cfg.topk_temperature
        self.max_budget = min(self.num_slots, cfg.budget_max)

        self.slot_embeddings = nn.Parameter(torch.randn(self.num_slots, self.slot_dim) * 0.02)
        self.query_proj = nn.Linear(cfg.policy_state_dim, self.slot_dim)

    def forward(self, x: torch.Tensor, budget: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, T, D]; budget: [B, T]. Returns (readout [B,T,slot_dim],
        weights [B,T,M] summing to 1 over each token's selected slots)."""
        q = self.query_proj(x)
        scores = torch.matmul(q, self.slot_embeddings.t()) / self.temperature  # [B,T,M]
        M = self.num_slots
        k = budget.round().long().clamp(min=1, max=self.max_budget)  # [B,T]
        # There is no need to sort or gather all M slots: AdaptiveBudget
        # guarantees that at most budget_max slots can be active.
        k_max = max(1, min(M, math.ceil(self.max_budget)))
        sorted_scores, idx = scores.topk(k_max, dim=-1)
        ranks = torch.arange(k_max, device=x.device).view(1, 1, k_max)
        selected = ranks < k.unsqueeze(-1)
        top_weights = F.softmax(sorted_scores.masked_fill(~selected, float("-inf")), dim=-1)
        slots = self.slot_embeddings[idx]  # [B,T,M,slot_dim]
        readout = (top_weights.unsqueeze(-1) * slots).sum(dim=2)
        # Preserve the public full-slot diagnostic tensor without paying for
        # full-slot sorting/gathering in the hot path.
        weights = torch.zeros_like(scores).scatter(-1, idx, top_weights)
        return readout, weights


class AdaptiveBudget(nn.Module):
    def __init__(self, cfg: SABERConfig):
        super().__init__()
        self.floor = cfg.budget_floor
        self.alpha = cfg.budget_alpha
        self.max_budget = cfg.budget_max

    def forward(self, surprise: torch.Tensor) -> torch.Tensor:
        budget = self.floor + self.alpha * surprise.clamp_min(0)
        return budget.clamp(self.floor, self.max_budget)


class SABER(nn.Module):
    """Surprise-adaptive memory layer: encoder → GRU policy → EMA predictor →
    surprise → budget → sparse slot readout, with an InfoNCE bottleneck loss
    and a JEPA-style predictor loss.
    """

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
        self, x: torch.Tensor, state: Optional[SABERState] = None
    ) -> tuple[SABERState, dict]:
        """x: [B, T, input_dim]. Returns (new_state, aux)."""
        if x.ndim != 3 or x.shape[1] == 0 or x.shape[2] != self.input_dim:
            raise ValueError(f"SABER expects [B,T,{self.input_dim}] with T>0, got {tuple(x.shape)}")
        B, T, _ = x.shape

        z = self.encoder(x)  # [B,T,D]
        hidden = state.policy_hidden if state is not None else None
        s, new_hidden = self.policy(hidden, z)  # [B,T,H]

        z_hat = self.predictor(s.detach(), use_ema=True)
        surprise = self.surprise(z, z_hat)
        budget = self.budget(surprise)
        readout, weights = self.memory(s, budget)  # slot queries from the policy state

        infonce_loss = self._compute_infonce(s, z)
        # Online predictor learns toward the latents (stop-grad both sides);
        # the EMA copy stays the stable surprise baseline.
        pred_online = self.predictor(s.detach(), use_ema=False)
        predictor_loss = F.mse_loss(pred_online, z.detach())

        if self.training:
            self._anneal_beta()
            self.step.add_(1)

        budget_history = list(state.budget_history) if state is not None else []
        infonce_history = list(state.infonce_loss_history) if state is not None else []
        budget_history.append(budget.mean().detach())
        infonce_history.append(infonce_loss.detach())

        new_state = SABERState(
            policy_hidden=new_hidden.detach(),
            budget_history=budget_history[-1000:],
            infonce_loss_history=infonce_history[-1000:],
        )
        aux = {
            "z_t": z,
            "s_t": s,
            "z_hat_t": z_hat,
            "surprise": surprise,
            "budget": budget,
            "infonce_loss": infonce_loss,
            "predictor_loss": predictor_loss,
            "memory_readout": readout,
            "memory_weights": weights,
        }
        return new_state, aux

    def update_ema(self):
        """Advance the predictor EMA. Called by the trainer after each step."""
        self.predictor.update_ema()

    def _compute_infonce(self, s_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """In-batch contrastive loss between time-averaged policy projections
        and encoder latents: logits[i, j] = ⟨proj(s_i), z_j⟩ / τ, target = i
        (all other batch elements act as negatives; no self-collision).  For
        batches larger than the configured negative budget, sample negatives
        without constructing a quadratic [B,B] matrix."""
        proj_s = self.policy.get_projection(s_t).mean(dim=1)  # [B,D]
        z_pooled = z_t.mean(dim=1)  # [B,D]
        B = proj_s.shape[0]
        n_neg = min(self.cfg.infonce_num_negatives, max(0, B - 1))
        positive = (proj_s * z_pooled).sum(dim=-1, keepdim=True)
        if n_neg == B - 1:
            logits = proj_s @ z_pooled.t()
            labels = torch.arange(B, device=logits.device)
        else:
            # Shift sampled indices at/after the row index so self is never a
            # negative.  Duplicate negatives are harmless and avoid a Python
            # loop or an O(B^2) permutation tensor.
            row = torch.arange(B, device=proj_s.device).unsqueeze(1)
            neg = torch.randint(B, (B, n_neg), device=proj_s.device)
            neg = neg + (neg >= row).long()
            negative = (proj_s.unsqueeze(1) * z_pooled[neg]).sum(dim=-1)
            logits = torch.cat([positive, negative], dim=1)
            labels = torch.zeros(B, dtype=torch.long, device=logits.device)
        logits = logits / self.cfg.infonce_temperature
        return F.cross_entropy(logits, labels)

    def _anneal_beta(self):
        # Keep the schedule on-device.  `int(self.step)` would synchronize a
        # CUDA scalar on every forward, which is particularly costly for
        # short streaming chunks.
        if self.cfg.infonce_beta_anneal_steps == 0:
            self.beta.copy_(self.beta.new_tensor(self.cfg.infonce_beta_end))
            return
        step = self.step.to(dtype=self.beta.dtype)
        total = self.beta.new_tensor(float(self.cfg.infonce_beta_anneal_steps))
        progress = (step / total).clamp(max=1.0)
        new_beta = self.beta.new_tensor(self.cfg.infonce_beta_start) * (1 - progress)
        new_beta = new_beta + self.beta.new_tensor(self.cfg.infonce_beta_end) * progress
        self.beta.copy_(new_beta)

    def get_param_groups(self) -> list[dict]:
        """Two-timescale groups + memory. Guaranteed disjoint and covering."""
        fast_params = (
            list(self.policy.parameters())
            + list(self.predictor.parameters())
            + list(self.surprise.parameters())
            + list(self.budget.parameters())
            + list(self.memory.query_proj.parameters())
        )
        slow_params = list(self.encoder.parameters())
        memory_params = [self.memory.slot_embeddings]
        return [
            {"params": fast_params, "lr": self.cfg.lr_fast, "name": "fast"},
            {"params": slow_params, "lr": self.cfg.lr_slow, "name": "slow"},
            {"params": memory_params, "lr": self.cfg.lr_memory, "name": "memory"},
        ]


class SABERBackbone(nn.Module):
    """Wrap a sequence backbone with SABER. With ``num_classes`` (and
    ``backbone_out_dim``) an additional mean-pool + linear head produces
    classification logits; otherwise the backbone hidden is returned.
    """

    def __init__(
        self,
        saber: SABER,
        backbone: nn.Module,
        write_gate_gamma: float = 1.0,
        num_classes: int | None = None,
        backbone_out_dim: int | None = None,
    ):
        super().__init__()
        if num_classes is not None and backbone_out_dim is None:
            raise ValueError("backbone_out_dim is required when num_classes is set")
        self.saber = saber
        self.backbone = backbone
        self.write_gate_gamma = write_gate_gamma
        self.head = nn.Linear(backbone_out_dim, num_classes) if num_classes is not None else None

    def forward(
        self,
        x: torch.Tensor,
        saber_state: Optional[SABERState] = None,
        backbone_states: Optional[list] = None,
    ) -> tuple[torch.Tensor, SABERState, list, dict]:
        new_saber_state, aux = self.saber(x, saber_state)
        write_strength = torch.sigmoid(self.write_gate_gamma * aux["surprise"])

        if self.training and hasattr(self.backbone, "forward_with_memory"):
            h, new_backbone_states = self.backbone.forward_with_memory(
                x, aux["memory_readout"], write_strength, backbone_states
            )
        else:
            h, new_backbone_states = self.backbone(x, backbone_states)

        y = self.head(h.mean(dim=1)) if self.head is not None else h
        return y, new_saber_state, new_backbone_states, aux
