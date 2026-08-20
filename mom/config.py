"""MoM configuration (spec §3, §4).

Single source of truth for the MoM architecture: expert bank, router,
masked-execution semantics and stability objectives.  Expert hyperparameters
default to the corresponding PRISM block values (§3.2).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import yaml

ROUTER_MODES = ("learned", "uniform", "random")


@dataclass
class MoMConfig:
    # Backbone shape
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4

    # Expert bank (§3.2). Names from mom.registry.EXPERT_NAMES.
    experts: tuple[str, ...] = ("ssd", "gdr")

    # Router (§3.3)
    top_k: int = 1
    router_mode: str = "learned"  # learned | uniform (B4) | random (B5)
    router_bias: bool = False  # optional input-independent b_e (default off)
    router_init_std: float = 0.01  # W_r ~ N(0, 0.01²) — near-uniform init
    router_seed: int = 0  # generator seed for router_mode="random"
    straight_through: bool = False  # ST gate estimator, R4 fallback
    router_surprise_scale: float = 0.0  # >0: add [B,T] surprise feature to logits

    # Shared expert (§3.7): one SSD instance always on, output added ungated.
    shared_expert: str | None = None  # None | "ssd"

    # Masked-execution semantics (§3.4).  D1: decay applies on every step
    # (default) or the state freezes on a miss.  For GDR the spec text
    # requires exact pass-through on a miss (its equations carry no α gate),
    # hence a separate default of False for PRISM's α forget gate.
    decay_on_skip: bool = True  # D1, applies to SSD's a_t
    gdr_decay_on_skip: bool = False  # applies to GDR's α_t (§3.4 pass-through)

    # Stability objectives (§3.7)
    lambda_bal: float = 1e-2
    lambda_z: float = 1e-3

    # Expert hyperparameters — PRISM defaults (§3.2)
    ssd_state_dim: int = 64
    s4_dt_min: float = 0.001
    s4_dt_max: float = 0.1
    delta_chunk_size: int = 64
    qk_norm: bool = True
    gate_bias_init: float = 4.0
    scan_backend: str = "auto"
    delta_backend: str = "reference"
    swa_window: int = 512  # v2

    # Shared block anatomy (PRISM-exact residual/pre-norm structure, §3.5)
    conv_kernel_size: int = 4
    ffn_expand: int = 2
    dropout: float = 0.0

    def __post_init__(self):
        if isinstance(self.experts, list):
            self.experts = tuple(self.experts)
        if not self.experts:
            raise ValueError("MoMConfig.experts must be non-empty")
        if len(set(self.experts)) != len(self.experts):
            raise ValueError(f"duplicate experts: {self.experts}")
        if self.router_mode not in ROUTER_MODES:
            raise ValueError(f"router_mode must be one of {ROUTER_MODES}")
        if not 1 <= self.top_k <= len(self.experts):
            raise ValueError(f"top_k must be in [1, {len(self.experts)}], got {self.top_k}")
        if self.shared_expert is not None and self.shared_expert != "ssd":
            raise ValueError("shared_expert must be None or 'ssd' (spec §3.7)")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.lambda_bal < 0 or self.lambda_z < 0:
            raise ValueError("loss weights must be non-negative")
        if self.router_surprise_scale < 0:
            raise ValueError("router_surprise_scale must be non-negative")
        if not (0 < self.s4_dt_min < self.s4_dt_max):
            raise ValueError("s4_dt_min must be < s4_dt_max")

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    def to_dict(self) -> dict:
        d = asdict(self)
        d["experts"] = list(self.experts)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MoMConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str) -> "MoMConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get("model", data))
