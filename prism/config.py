from __future__ import annotations

from dataclasses import dataclass, field

# Valid per-layer role tokens.
#   "s4"    — SSM sequence-mixer slot. Concrete implementation is chosen by
#             `ssm_kind` ("ssd" → Mamba-2-style selective scan, "s4d_legacy" → S4D-Complex).
#   "delta" — Gated Delta Rule (matrix-valued associative memory).
#   "swa"   — Sliding-window attention (for H1-style hybrid ablations).
LAYER_TOKENS = ("s4", "delta", "swa")


@dataclass
class ModalityConfig:
    name: str
    input_dim: int
    num_classes: int
    patch_size: int | None = None
    window_size: int | None = None
    # Multi-label task (e.g. PTB-XL, where a record can carry several diagnostic
    # superclasses). Selects BCEWithLogits loss + multi-hot targets + macro AUROC
    # instead of softmax cross-entropy + argmax accuracy.
    multilabel: bool = False


@dataclass
class PRISMConfig:
    # Model boyutu
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 12

    # Blok dağılımı (legacy interleave): her delta_every. katman "delta".
    delta_every: int = 4

    # Explicit per-layer pattern. When set, overrides delta_every / force_block_type.
    # Either a list of tokens (["s4", "s4", "s4", "delta", ...]) or a comma/space
    # separated string ("s4,s4,s4,delta"). Length must equal num_layers.
    block_pattern: list[str] | str | None = None

    # SSM implementation for the "s4" role.
    #   "ssd"        — Mamba-2-style SSD: scalar-per-head decay, per-channel state,
    #                  input-dependent (selective) Δ/B/C. No mean-over-Dh collapse.
    #   "s4d_legacy" — original diagonal S4D-Complex block (kept as ablation row).
    ssm_kind: str = "ssd"

    # S4D-legacy init: "lin" (S4D-Lin, A_n = -1/2 + iπn) or "legacy" (the original
    # linspace/π·arange init). Only used when ssm_kind == "s4d_legacy".
    s4d_init: str = "lin"

    # S4 / SSD parametreleri
    s4_state_mult: int = 2
    s4_dt_min: float = 0.001
    s4_dt_max: float = 0.1
    ssd_state_dim: int = 64  # N for the SSD block (state dim per head, shared over Dh)

    # Delta parametreleri
    delta_chunk_size: int = 64
    qk_norm: bool = True
    gate_bias_init: float = 4.0
    # Delta-rule backend: "reference" (pure-PyTorch chunked solve, always available)
    # or "fla" (flash-linear-attention Triton kernel, GPU-only, falls back if absent).
    delta_backend: str = "reference"

    # Scan backend for the SSD/S4D linear recurrence:
    #   "auto"      — torch.associative_scan if available, else reference.
    #   "assoc"     — force torch.associative_scan.
    #   "reference" — vectorized Hillis-Steele scan (no torch HOP, always correct).
    scan_backend: str = "auto"

    # Sliding-window attention (used by "swa" layers).
    swa_window: int = 128

    # Shared
    conv_kernel_size: int = 4
    ffn_expand: int = 2

    # Dropout applied after mixer/attention and FFN sublayers, and before the
    # classification head. 0.0 disables dropout (default for the research baseline).
    dropout: float = 0.0

    # torch.compile the backbone (trainer reads this flag).
    compile: bool = False

    # Modaliteler
    modalities: list[ModalityConfig] = field(default_factory=list)

    # Pooling stratejisi: "mean" veya "last"
    pool_type: str = "mean"

    # Ablation: None = normal S4/Delta interleave; "s4" | "delta" = all layers that type.
    force_block_type: str | None = None

    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0, (
            f"hidden_dim {self.hidden_dim} must be divisible by num_heads {self.num_heads}"
        )
        assert self.ssm_kind in ("ssd", "s4d_legacy"), (
            f"ssm_kind must be 'ssd' or 's4d_legacy', got {self.ssm_kind!r}"
        )
        assert self.s4d_init in ("lin", "legacy"), (
            f"s4d_init must be 'lin' or 'legacy', got {self.s4d_init!r}"
        )
        assert self.delta_backend in ("reference", "fla"), (
            f"delta_backend must be 'reference' or 'fla', got {self.delta_backend!r}"
        )
        assert self.scan_backend in ("auto", "assoc", "reference"), (
            f"scan_backend must be 'auto', 'assoc', or 'reference', got {self.scan_backend!r}"
        )
        # Positive dimensional hyperparameters.
        for name in (
            "hidden_dim",
            "num_heads",
            "num_layers",
            "s4_state_mult",
            "ssd_state_dim",
            "delta_chunk_size",
            "conv_kernel_size",
            "ffn_expand",
            "swa_window",
        ):
            val = getattr(self, name)
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"{name} must be a positive integer, got {val!r}")
        if not (0 < self.s4_dt_min < self.s4_dt_max):
            raise ValueError(
                f"s4_dt_min must be positive and less than s4_dt_max, "
                f"got dt_min={self.s4_dt_min}, dt_max={self.s4_dt_max}"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        # Modality sanity checks.
        if not self.modalities:
            raise ValueError("PRISMConfig.modalities must contain at least one ModalityConfig")
        modality_names = [m.name for m in self.modalities]
        if len(modality_names) != len(set(modality_names)):
            raise ValueError(f"Modality names must be unique, got {modality_names}")
        for m in self.modalities:
            if m.input_dim <= 0:
                raise ValueError(
                    f"Modality '{m.name}' input_dim must be positive, got {m.input_dim}"
                )
            if m.num_classes <= 0:
                raise ValueError(
                    f"Modality '{m.name}' num_classes must be positive, got {m.num_classes}"
                )
        # Normalize an explicit block_pattern string → list[str].
        if isinstance(self.block_pattern, str):
            self.block_pattern = [
                tok for tok in self.block_pattern.replace(",", " ").split() if tok
            ]
        if self.block_pattern is not None and self.force_block_type is not None:
            raise ValueError(
                "block_pattern and force_block_type cannot both be set; "
                "use block_pattern for an explicit per-layer pattern or "
                "force_block_type for an all-one-type ablation."
            )
        if self.block_pattern is not None:
            bad = [t for t in self.block_pattern if t not in LAYER_TOKENS]
            if bad:
                raise ValueError(
                    f"block_pattern has unknown tokens {bad}; allowed: {list(LAYER_TOKENS)}"
                )
            if len(self.block_pattern) != self.num_layers:
                raise ValueError(
                    f"block_pattern length {len(self.block_pattern)} != num_layers "
                    f"{self.num_layers}"
                )
        elif self.force_block_type is None:
            if self.delta_every <= 0:
                raise ValueError(f"delta_every must be positive, got {self.delta_every}")
        else:
            if self.force_block_type not in ("s4", "delta"):
                raise ValueError(
                    f"force_block_type must be 's4', 'delta', or None, got {self.force_block_type!r}"
                )
        if self.pool_type not in ("mean", "last"):
            raise ValueError(f"pool_type must be 'mean' or 'last', got {self.pool_type!r}")

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def s4_state_dim(self) -> int:
        return self.head_dim * self.s4_state_mult

    def layer_type(self, i: int) -> str:
        """i. katmanın rol token'ı: 's4', 'delta' veya 'swa'."""
        if self.block_pattern is not None:
            return self.block_pattern[i]
        if self.force_block_type is not None:
            return self.force_block_type
        if (i + 1) % self.delta_every == 0:
            return "delta"
        return "s4"

    def layer_pattern(self) -> list[str]:
        return [self.layer_type(i) for i in range(self.num_layers)]
