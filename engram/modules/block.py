from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

import torch
import torch.nn as nn

from engram.layer_tokens import LAYER_TOKENS

from .attention import SWABlock, SWAState
from .delta import DeltaBlock, DeltaState
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


def _build_s4_block(cfg: ENGRAMConfig) -> nn.Module:
    """Resolve the ``s4`` role to SSD or S4D based on ``cfg.ssm_kind``."""
    common = dict(
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        conv_kernel_size=cfg.conv_kernel_size,
        ffn_expand=cfg.ffn_expand,
        dropout=cfg.dropout,
    )
    if cfg.ssm_kind == "ssd":
        return SSDBlock(
            **common,
            state_dim=cfg.ssd_state_dim,
            dt_min=cfg.s4_dt_min,
            dt_max=cfg.s4_dt_max,
            scan_backend=cfg.scan_backend,
        )
    return S4Block(
        **common,
        state_mult=cfg.s4_state_mult,
        dt_min=cfg.s4_dt_min,
        dt_max=cfg.s4_dt_max,
        init=cfg.s4d_init,
        scan_backend=cfg.scan_backend,
    )


@register_block("s4")
def _build_s4(cfg: ENGRAMConfig) -> nn.Module:
    return _build_s4_block(cfg)


@register_block("delta")
def _build_delta(cfg: ENGRAMConfig) -> nn.Module:
    return DeltaBlock(
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        qk_norm=cfg.qk_norm,
        chunk_size=cfg.delta_chunk_size,
        gate_bias_init=cfg.gate_bias_init,
        conv_kernel_size=cfg.conv_kernel_size,
        ffn_expand=cfg.ffn_expand,
        backend=cfg.delta_backend,
        dropout=cfg.dropout,
    )


@register_block("swa")
def _build_swa(cfg: ENGRAMConfig) -> nn.Module:
    return SWABlock(
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        window=cfg.swa_window,
        ffn_expand=cfg.ffn_expand,
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
    """Type-agnostic block forward — ``ENGRAMBackbone`` calls this.

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
