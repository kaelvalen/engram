from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SABERConfig
from .saber import SABER

logger = logging.getLogger(__name__)


@dataclass
class SABERDiagnostics:
    budget_history: list[float] = field(default_factory=list)
    budget_var_history: list[float] = field(default_factory=list)
    infonce_loss_history: list[float] = field(default_factory=list)
    policy_grad_norm_history: list[float] = field(default_factory=list)
    backbone_grad_norm_history: list[float] = field(default_factory=list)
    memory_grad_norm_history: list[float] = field(default_factory=list)

    r1_triggered: int = 0
    r2_triggered: int = 0
    r3_triggered: int = 0
    r4_triggered: int = 0
    r5_triggered: int = 0

    def update(
        self,
        budget: torch.Tensor,
        infonce_loss: torch.Tensor,
        policy_grad_norm: Optional[float] = None,
        backbone_grad_norm: Optional[float] = None,
        memory_grad_norm: Optional[float] = None,
    ):
        self.budget_history.append(budget.mean().item())
        self.budget_var_history.append(budget.var().item())
        self.infonce_loss_history.append(infonce_loss.item())

        if policy_grad_norm is not None:
            self.policy_grad_norm_history.append(policy_grad_norm)
        if backbone_grad_norm is not None:
            self.backbone_grad_norm_history.append(backbone_grad_norm)
        if memory_grad_norm is not None:
            self.memory_grad_norm_history.append(memory_grad_norm)

        if len(self.budget_history) > 1000:
            self.budget_history = self.budget_history[-1000:]
            self.budget_var_history = self.budget_var_history[-1000:]
            self.infonce_loss_history = self.infonce_loss_history[-1000:]
            self.policy_grad_norm_history = self.policy_grad_norm_history[-1000:]
            self.backbone_grad_norm_history = self.backbone_grad_norm_history[-1000:]
            self.memory_grad_norm_history = self.memory_grad_norm_history[-1000:]

    def check_r1_predictor_dominance(self, cfg: SABERConfig) -> bool:
        if len(self.budget_history) < cfg.r1_patience:
            return False
        recent_budget = self.budget_history[-cfg.r1_patience :]
        avg_budget = sum(recent_budget) / len(recent_budget)
        return avg_budget < cfg.r1_budget_threshold * cfg.budget_floor

    def check_r2_memory_dominance(self, cfg: SABERConfig) -> bool:
        if (
            len(self.memory_grad_norm_history) < cfg.r2_patience
            or len(self.backbone_grad_norm_history) < cfg.r2_patience
        ):
            return False
        recent_mem = self.memory_grad_norm_history[-cfg.r2_patience :]
        recent_backbone = self.backbone_grad_norm_history[-cfg.r2_patience :]
        avg_mem = sum(recent_mem) / len(recent_mem)
        avg_backbone = sum(recent_backbone) / len(recent_backbone)
        return avg_mem > cfg.r2_grad_ratio_threshold * avg_backbone

    def check_r3_noisy_overactivation(self, cfg: SABERConfig) -> bool:
        if len(self.budget_var_history) < cfg.r3_patience:
            return False
        recent_var = self.budget_var_history[-cfg.r3_patience :]
        avg_var = sum(recent_var) / len(recent_var)
        threshold = (cfg.budget_alpha * cfg.r3_var_threshold_multiplier) ** 2
        return avg_var > threshold

    def check_r4_compute_starvation(self, cfg: SABERConfig) -> bool:
        if len(self.budget_history) < cfg.r4_patience:
            return False
        recent_budget = self.budget_history[-cfg.r4_patience :]
        avg_budget = sum(recent_budget) / len(recent_budget)
        threshold = cfg.budget_floor + cfg.r4_budget_threshold * cfg.budget_alpha
        return avg_budget < threshold

    def check_r5_policy_backbone_emergence(self, cfg: SABERConfig) -> bool:
        if len(self.infonce_loss_history) < 100:
            return False
        recent_infonce = self.infonce_loss_history[-100:]
        avg_infonce = sum(recent_infonce) / len(recent_infonce)
        return avg_infonce < cfg.r5_infonce_threshold


class SABERRecovery:
    def __init__(self, saber: SABER, cfg: SABERConfig):
        self.saber = saber
        self.cfg = cfg
        self.diagnostics = SABERDiagnostics()

    def check_and_recover(self, aux: dict, step: int) -> list[str]:
        triggered = []

        if self.diagnostics.check_r1_predictor_dominance(self.cfg):
            self._recover_r1()
            triggered.append("R1")
            self.diagnostics.r1_triggered += 1
            logger.warning(f"R1 triggered at step {step}: Predictor Dominance")

        if self.diagnostics.check_r2_memory_dominance(self.cfg):
            self._recover_r2()
            triggered.append("R2")
            self.diagnostics.r2_triggered += 1
            logger.warning(f"R2 triggered at step {step}: Memory Dominance")

        if self.diagnostics.check_r3_noisy_overactivation(self.cfg):
            self._recover_r3()
            triggered.append("R3")
            self.diagnostics.r3_triggered += 1
            logger.warning(f"R3 triggered at step {step}: Noisy Over-Activation")

        if self.diagnostics.check_r4_compute_starvation(self.cfg):
            self._recover_r4()
            triggered.append("R4")
            self.diagnostics.r4_triggered += 1
            logger.warning(f"R4 triggered at step {step}: Compute Starvation")

        if self.diagnostics.check_r5_policy_backbone_emergence(self.cfg):
            self._recover_r5()
            triggered.append("R5")
            self.diagnostics.r5_triggered += 1
            logger.warning(f"R5 triggered at step {step}: Policy Backbone Emergence")

        return triggered

    def _recover_r1(self):
        nn.init.normal_(self.saber.predictor.ema_weight, mean=0.0, std=0.02)
        self.saber.predictor.linear.weight.data.copy_(self.saber.predictor.ema_weight)

        with torch.no_grad():
            noise = torch.randn_like(self.saber.predictor.ema_weight) * 0.1
            self.saber.predictor.ema_weight.add_(noise)

        self.cfg.budget_floor = int(self.cfg.budget_floor * 1.2)

    def _recover_r2(self):
        self.saber.memory.slot_embeddings.grad = None
        self.saber.memory.slot_embeddings.requires_grad_(False)

    def _recover_r3(self):
        self.saber.surprise.eps_scale *= 1.5
        self.cfg.surprise_eps_scale = self.saber.surprise.eps_scale

    def _recover_r4(self):
        self.cfg.infonce_beta_end *= 0.5
        self.cfg.budget_alpha *= 1.25

        with torch.no_grad():
            noise = torch.randn_like(self.saber.encoder.net[-1].weight) * 0.01
            self.saber.encoder.net[-1].weight.add_(noise)

    def _recover_r5(self):
        # Reset the policy IN PLACE: replacing the module would orphan the
        # optimizer's parameter references.
        with torch.no_grad():
            for p in self.saber.policy.parameters():
                if p.dim() > 1:
                    nn.init.normal_(p, mean=0.0, std=0.02)
                else:
                    nn.init.zeros_(p)
        self.cfg.policy_state_dim = min(self.cfg.policy_state_dim, 64)
        self.saber.beta.data = torch.tensor(self.cfg.infonce_beta_start)


class SABERTrainer:
    def __init__(
        self,
        saber_backbone: nn.Module,
        cfg: SABERConfig,
        device: torch.device,
    ):
        self.saber_backbone = saber_backbone
        self.cfg = cfg
        self.device = device
        self.step = 0
        self.phase = 1

        self.recovery = SABERRecovery(saber_backbone.saber, cfg)

        # One optimizer over saber param groups PLUS the backbone/head —
        # otherwise phase-1 task training has nothing to step.
        param_groups = saber_backbone.saber.get_param_groups()
        backbone_params = list(saber_backbone.backbone.parameters())
        if saber_backbone.head is not None:
            backbone_params += list(saber_backbone.head.parameters())
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": cfg.lr_slow, "name": "backbone"})
        self.optimizer = torch.optim.AdamW(param_groups)

    def get_phase(self) -> int:
        if self.step < self.cfg.phase1_steps:
            return 1
        elif self.step < self.cfg.phase1_steps + self.cfg.phase2_steps:
            return 2
        return 3

    def train_step(self, batch: dict) -> dict:
        new_phase = self.get_phase()
        if new_phase != self.phase:
            self.phase = new_phase
            logger.info(f"Entering Phase {self.phase} at step {self.step + 1}")

        # Phase is decided by steps already completed, so each phase runs
        # exactly its configured number of steps.
        self.step += 1
        if self.phase == 1:
            return self._phase1_step(batch)
        elif self.phase == 2:
            return self._phase2_step(batch)
        else:
            return self._phase3_step(batch)

    def _phase1_step(self, batch: dict) -> dict:
        for param in self.saber_backbone.saber.policy.parameters():
            param.requires_grad_(False)
        for param in self.saber_backbone.saber.predictor.parameters():
            param.requires_grad_(False)
        for param in self.saber_backbone.saber.surprise.parameters():
            param.requires_grad_(False)
        for param in self.saber_backbone.saber.budget.parameters():
            param.requires_grad_(False)
        for param in self.saber_backbone.saber.memory.parameters():
            param.requires_grad_(False)

        x = batch["x"].to(self.device)
        labels = batch["labels"].to(self.device)

        y, _, _, _ = self.saber_backbone(x)
        loss = F.cross_entropy(y, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "phase": 1}

    def _phase2_step(self, batch: dict) -> dict:
        for param in self.saber_backbone.backbone.parameters():
            param.requires_grad_(False)

        for param in self.saber_backbone.saber.parameters():
            param.requires_grad_(True)

        x = batch["x"].to(self.device)
        labels = batch["labels"].to(self.device)

        y, saber_state, _, aux = self.saber_backbone(x)
        task_loss = F.cross_entropy(y, labels)
        infonce_loss = aux["infonce_loss"]
        predictor_loss = aux["predictor_loss"]
        loss = (
            task_loss
            + self.saber_backbone.saber.beta * infonce_loss
            + self.cfg.predictor_loss_weight * predictor_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.saber_backbone.saber.update_ema()

        self.recovery.diagnostics.update(
            aux["budget"],
            infonce_loss,
        )

        return {
            "loss": loss.item(),
            "task_loss": task_loss.item(),
            "infonce_loss": infonce_loss.item(),
            "predictor_loss": predictor_loss.item(),
            "phase": 2,
        }

    def _phase3_step(self, batch: dict) -> dict:
        for param in self.saber_backbone.parameters():
            param.requires_grad_(True)

        x = batch["x"].to(self.device)
        labels = batch["labels"].to(self.device)

        y, saber_state, _, aux = self.saber_backbone(x)
        task_loss = F.cross_entropy(y, labels)
        infonce_loss = aux["infonce_loss"]
        predictor_loss = aux["predictor_loss"]
        loss = (
            task_loss
            + self.saber_backbone.saber.beta * infonce_loss
            + self.cfg.predictor_loss_weight * predictor_loss
        )

        self.optimizer.zero_grad()
        loss.backward()

        policy_grad = sum(
            p.grad.norm().item()
            for p in self.saber_backbone.saber.policy.parameters()
            if p.grad is not None
        )
        backbone_grad = sum(
            p.grad.norm().item()
            for p in self.saber_backbone.backbone.parameters()
            if p.grad is not None
        )
        memory_grad = sum(
            p.grad.norm().item()
            for p in self.saber_backbone.saber.memory.parameters()
            if p.grad is not None
        )

        self.optimizer.step()
        self.saber_backbone.saber.update_ema()

        self.recovery.diagnostics.update(
            aux["budget"],
            infonce_loss,
            policy_grad,
            backbone_grad,
            memory_grad,
        )

        triggered = self.recovery.check_and_recover(aux, self.step)

        if self.step % self.cfg.log_every == 0:
            logger.info(
                f"Step {self.step}: loss={loss.item():.4f}, "
                f"budget={aux['budget'].mean().item():.2f}, "
                f"infonce={infonce_loss.item():.4f}, "
                f"triggered={triggered}"
            )

        return {
            "loss": loss.item(),
            "task_loss": task_loss.item(),
            "infonce_loss": infonce_loss.item(),
            "predictor_loss": predictor_loss.item(),
            "budget": aux["budget"].mean().item(),
            "surprise": aux["surprise"].mean().item(),
            "phase": 3,
            "recovery_triggered": triggered,
        }
