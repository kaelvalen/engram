from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SABERConfig:
    # Latent encoder
    encoder_hidden_dim: int = 256
    encoder_num_layers: int = 2
    encoder_dropout: float = 0.1

    # Policy state (intentionally capacity-limited)
    policy_state_dim: int = 64
    policy_num_layers: int = 2
    policy_dropout: float = 0.1

    # InfoNCE bottleneck
    infonce_temperature: float = 0.1
    infonce_num_negatives: int = 128
    infonce_beta_start: float = 0.01
    infonce_beta_end: float = 1.0
    infonce_beta_anneal_steps: int = 10000

    # Predictor (hardened)
    predictor_ema_decay: float = 0.999
    predictor_max_steps: int = 100000
    predictor_loss_weight: float = 1.0  # weight of the JEPA-style predictor loss

    # Surprise estimator
    surprise_mu_lambda: float = 0.99
    surprise_sigma_lambda: float = 0.99
    surprise_eps_min: float = 1e-4
    surprise_eps_scale: float = 1.0
    surprise_max: float = 10.0

    # Adaptive compute budget
    budget_floor: int = 4
    budget_alpha: float = 2.0
    budget_max: int = 32

    # Memory slots (NOESIS integration)
    num_memory_slots: int = 64
    memory_slot_dim: int = 256

    # Sparse activation
    topk_temperature: float = 1.0

    # Write gating
    write_gate_gamma: float = 1.0

    # Recovery thresholds
    r1_budget_threshold: float = 0.5
    r1_patience: int = 100
    r2_grad_ratio_threshold: float = 5.0
    r2_patience: int = 200
    r3_var_threshold_multiplier: float = 1.0
    r3_patience: int = 100
    r4_budget_threshold: float = 0.1
    r4_patience: int = 500
    r5_infonce_threshold: float = 0.1

    # Two-timescale learning rates
    lr_fast: float = 1e-3
    lr_slow: float = 1e-4
    lr_memory: float = 5e-4

    # Training phases
    phase1_steps: int = 10000
    phase2_steps: int = 10000
    phase3_steps: int = 100000

    # Diagnostics
    log_every: int = 100

    def __post_init__(self):
        assert self.policy_state_dim <= 64, "Policy state dim must be <= 64"
        assert self.budget_floor >= 1
        assert self.budget_max >= self.budget_floor
        assert self.num_memory_slots >= self.budget_max
        assert 0 < self.predictor_ema_decay < 1
        assert 0 < self.surprise_mu_lambda < 1
        assert 0 < self.surprise_sigma_lambda < 1
        assert self.infonce_beta_start <= self.infonce_beta_end
