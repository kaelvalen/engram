from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModalityConfig:
    name: str
    input_dim: int
    num_classes: int
    patch_size: int | None = None
    window_size: int | None = None


@dataclass
class PRISMConfig:
    # Model boyutu
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 12

    # Blok dağılımı
    delta_every: int = 4

    # S4 parametreleri
    s4_state_mult: int = 2
    s4_dt_min: float = 0.001
    s4_dt_max: float = 0.1

    # Delta parametreleri
    delta_chunk_size: int = 64
    qk_norm: bool = True
    gate_bias_init: float = 4.0

    # Shared
    conv_kernel_size: int = 4
    ffn_expand: int = 2

    # Modaliteler
    modalities: list[ModalityConfig] = field(default_factory=list)

    # Pooling stratejisi: "mean" veya "last"
    pool_type: str = "mean"

    # Ablation: None = normal S4/Delta interleave; "s4" | "delta" = all layers that type
    force_block_type: str | None = None

    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0, (
            f"hidden_dim {self.hidden_dim} must be divisible by num_heads {self.num_heads}"
        )
        if self.force_block_type is None:
            assert self.delta_every > 0, f"delta_every must be positive, got {self.delta_every}"
        else:
            assert self.force_block_type in ("s4", "delta"), (
                f"force_block_type must be 's4', 'delta', or None, got {self.force_block_type!r}"
            )
        assert self.pool_type in ["mean", "last"], f"Unknown pool_type: {self.pool_type}"

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def s4_state_dim(self) -> int:
        return self.head_dim * self.s4_state_mult

    def layer_type(self, i: int) -> str:
        """i. katmanın tipi: 's4' veya 'delta'"""
        if self.force_block_type is not None:
            return self.force_block_type
        if (i + 1) % self.delta_every == 0:
            return "delta"
        return "s4"

    def layer_pattern(self) -> list[str]:
        return [self.layer_type(i) for i in range(self.num_layers)]
