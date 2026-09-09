from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

import torch
import torch.nn as nn

from engram.layer_tokens import LAYER_TOKENS

from .attention import SWABlock, SWAState
from .conv import ShortCausalConv1d
from .delta import DeltaBlock, DeltaState
from .ffn import SwiGLU
from .norm import RMSNorm
from .s4 import S4Block
from .ssd import SSDBlock

if TYPE_CHECKING:
    from engram.config import ENGRAMConfig

__all__ = [
    "LAYER_TOKENS",
    "BlockState",
    "SSDBlockState",
    "DeltaBlockState",
    "SWABlockState",
    "ENGRAMBlock",
    "BlockBuilder",
    "BLOCK_REGISTRY",
    "register_block",
    "build_block",
]


@dataclass
class BlockState:
    """Carry state for a single ENGRAM block.

    Subclasses add typed accessors for the concrete mixer state returned by each
    block family (SSM tensor, Delta matrix state, or attention KV-cache).
    """

    conv_state: torch.Tensor | None  # [B, dim, kernel_size-1] (None for attention)
    mixer_state: torch.Tensor | DeltaState | None


@dataclass
class SSDBlockState(BlockState):
    """State carried by SSD/S4D blocks."""

    mixer_state: torch.Tensor | None


@dataclass
class DeltaBlockState(BlockState):
    """State carried by Gated Delta Rule blocks."""

    mixer_state: DeltaState | None


@dataclass
class SWABlockState(BlockState):
    """State carried by sliding-window attention blocks.

    ``mixer_state`` is the attention KV-cache (last ``window`` RoPE-applied
    keys/values + absolute position); ``conv_state`` is always None (SWA
    blocks have no conv).
    """

    mixer_state: SWAState | None = None


@runtime_checkable
class ENGRAMBlock(Protocol):
    """Protocol that every ENGRAM residual block must satisfy.

    Blocks accept the residual input and optional carried states, and return the
    updated residual plus new conv/mixer states.  The exact type of the mixer
    state is block-family specific, hence the loose return annotation; callers
    should rely on the per-family ``BlockState`` subclasses for typing.
    """

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        mixer_state: torch.Tensor | DeltaState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | DeltaState | None]: ...


BlockBuilder = Callable[["ENGRAMConfig"], nn.Module]

# Registry mapping per-layer role tokens to builder callables.  The ``s4`` token
# resolves to SSD or S4D depending on ``ENGRAMConfig.ssm_kind``, keeping the
# public ``block_pattern`` API unchanged.
BLOCK_REGISTRY: dict[str, BlockBuilder] = {}


def register_block(
    token: str, builder: BlockBuilder | None = None
) -> BlockBuilder | Callable[[BlockBuilder], BlockBuilder]:
    """Register a block builder under ``token``.

    May be used as a decorator:

        @register_block("my_block")
        def build_my_block(cfg: ENGRAMConfig) -> nn.Module:
            ...

    or as a direct call:

        register_block("my_block", build_my_block)
    """

    def _register(fn: BlockBuilder) -> BlockBuilder:
        if token in BLOCK_REGISTRY:
            raise ValueError(f"Block token {token!r} is already registered")
        BLOCK_REGISTRY[token] = fn
        return fn

    if builder is not None:
        return _register(builder)
    return _register


def _maybe_conv(hidden_dim: int, kernel_size: int):
    """Return a conv module or None (ablation bypass) when kernel_size <= 0."""
    if kernel_size <= 0:
        return None
    return ShortCausalConv1d(hidden_dim, kernel_size)


def _maybe_ffn(hidden_dim: int, expand: int, dropout: float = 0.0):
    """Return (norm, ffn, dropout) or None when expand <= 0 (ablation bypass)."""
    if expand <= 0:
        return None, None, None
    return (
        RMSNorm(hidden_dim),
        SwiGLU(hidden_dim, expand),
        nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
    )


class _AblationBlockWrapper(nn.Module):
    """Wraps a standard block to support conv/FFN bypass for component ablation.

    When conv_kernel_size=0 or ffn_expand=0 in the config, the corresponding
    component is removed from the block. This wrapper replaces the original
    block's conv/FFN with identity operations by intercepting the forward.
    """

    def __init__(self, inner: nn.Module, skip_conv: bool, skip_ffn: bool):
        super().__init__()
        self.inner = inner
        self.skip_conv = skip_conv
        self.skip_ffn = skip_ffn

        if skip_conv and hasattr(inner, "conv"):
            # Replace conv with identity-like passthrough
            inner.conv = None  # type: ignore[assignment]

        if skip_ffn:
            if hasattr(inner, "ffn"):
                inner.ffn = None  # type: ignore[assignment]
            if hasattr(inner, "norm2"):
                inner.norm2 = None  # type: ignore[assignment]
            if hasattr(inner, "ffn_dropout"):
                inner.ffn_dropout = None  # type: ignore[assignment]

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        mixer_state=None,
    ):
        r = x
        x_n = self.inner.norm1(x)

        # Conv or bypass
        if self.skip_conv:
            x_c = x_n
            new_conv_state = None
        else:
            x_c, new_conv_state = self.inner.conv(x_n, conv_state)

        # Mixer (SSM / Delta / etc.)
        if hasattr(self.inner, "ssm"):
            x_m, new_mixer_state = self.inner.ssm(x_c, mixer_state)
        elif hasattr(self.inner, "delta"):
            x_m, new_mixer_state = self.inner.delta(x_c, mixer_state)
        else:
            raise AttributeError("Block has no recognized mixer (ssm/delta)")

        drop = getattr(self.inner, "dropout", nn.Identity())
        x = r + drop(x_m)

        # FFN or bypass
        if not self.skip_ffn and self.inner.norm2 is not None:
            ffn_drop = getattr(self.inner, "ffn_dropout", nn.Identity())
            x = x + ffn_drop(self.inner.ffn(self.inner.norm2(x)))

        return x, new_conv_state, new_mixer_state


def _wrap_if_needed(block: nn.Module, cfg: ENGRAMConfig) -> nn.Module:
    """Wrap a block with ablation bypass if conv or FFN are disabled."""
    skip_conv = cfg.conv_kernel_size <= 0
    skip_ffn = cfg.ffn_expand <= 0
    if skip_conv or skip_ffn:
        return _AblationBlockWrapper(block, skip_conv, skip_ffn)
    return block


def _build_s4_block(cfg: ENGRAMConfig) -> nn.Module:
    """Resolve the ``s4`` role to SSD or S4D based on ``cfg.ssm_kind``."""
    # Use kernel_size >= 1 for the actual block; ablation wrapping handles bypass.
    conv_ks = max(cfg.conv_kernel_size, 1)
    ffn_exp = max(cfg.ffn_expand, 1)
    common = dict(
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        conv_kernel_size=conv_ks,
        ffn_expand=ffn_exp,
        dropout=cfg.dropout,
    )
    if cfg.ssm_kind == "ssd":
        block = SSDBlock(
            **common,
            state_dim=cfg.ssd_state_dim,
            dt_min=cfg.s4_dt_min,
            dt_max=cfg.s4_dt_max,
            scan_backend=cfg.scan_backend,
        )
    else:
        block = S4Block(
            **common,
            state_mult=cfg.s4_state_mult,
            dt_min=cfg.s4_dt_min,
            dt_max=cfg.s4_dt_max,
            init=cfg.s4d_init,
            scan_backend=cfg.scan_backend,
        )
    return _wrap_if_needed(block, cfg)


@register_block("s4")
def _build_s4(cfg: ENGRAMConfig) -> nn.Module:
    return _build_s4_block(cfg)


@register_block("delta")
def _build_delta(cfg: ENGRAMConfig) -> nn.Module:
    conv_ks = max(cfg.conv_kernel_size, 1)
    ffn_exp = max(cfg.ffn_expand, 1)
    block = DeltaBlock(
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        qk_norm=cfg.qk_norm,
        chunk_size=cfg.delta_chunk_size,
        gate_bias_init=cfg.gate_bias_init,
        conv_kernel_size=conv_ks,
        ffn_expand=ffn_exp,
        backend=cfg.delta_backend,
        dropout=cfg.dropout,
        out_gate=cfg.delta_out_gate,
        memoryless=cfg.delta_memoryless,
    )
    return _wrap_if_needed(block, cfg)


@register_block("swa")
def _build_swa(cfg: ENGRAMConfig) -> nn.Module:
    return SWABlock(
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        window=cfg.swa_window,
        ffn_expand=max(cfg.ffn_expand, 1),
        dropout=cfg.dropout,
    )


def build_block(layer_type: str, cfg: ENGRAMConfig) -> nn.Module:
    """Build a block from config using the ``BLOCK_REGISTRY``.

    Raises:
        ValueError: if ``layer_type`` is not a registered block token.
    """
    builder = BLOCK_REGISTRY.get(layer_type)
    if builder is None:
        raise ValueError(
            f"Unknown layer type: {layer_type!r}. Registered tokens: {list(BLOCK_REGISTRY)}"
        )
    return builder(cfg)


def forward_block(
    block: nn.Module,
    layer_type: str,
    x: torch.Tensor,
    state: BlockState | None,
) -> tuple[torch.Tensor, BlockState]:
    """Type-agnostic block forward - ``ENGRAMBackbone`` calls this.

    All blocks (S4/SSD/Delta/SWA) share the same signature:
        (x, conv_state, mixer_state) -> (x, new_conv_state, new_mixer_state)
    """
    conv_state = state.conv_state if state is not None else None
    mixer_state = state.mixer_state if state is not None else None
    x, new_conv, new_mixer = block(x, conv_state, mixer_state)

    if layer_type == "delta":
        new_state: BlockState = DeltaBlockState(
            conv_state=new_conv,
            mixer_state=new_mixer,  # type: ignore[arg-type]
        )
    elif layer_type == "swa":
        new_state = SWABlockState(conv_state=new_conv, mixer_state=new_mixer)
    else:
        new_state = SSDBlockState(conv_state=new_conv, mixer_state=new_mixer)  # type: ignore[arg-type]
    return x, new_state
