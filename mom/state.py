"""Streaming state container (spec §3.6).

Global state = { (ℓ, e) → State }, keyed by layer index and expert name.
Two reserved non-expert keys carry the block's conv state and the optional
always-on shared expert's state.
"""

from __future__ import annotations

import torch
from prism.modules.delta import DeltaState

CONV_KEY = "__conv__"
SHARED_KEY = "__shared__"

StateKey = tuple[int, str]
MixerState = "torch.Tensor | DeltaState"


def _detach(state):
    if isinstance(state, torch.Tensor):
        return state.detach()
    if hasattr(state, "detach"):
        return state.detach()  # DeltaState, SWAState
    raise TypeError(f"unsupported state type: {type(state)!r}")


class ExpertStateDict(dict):
    """dict[(layer_idx, expert_name) -> mixer state] with detach support.

    Keys are ``(layer_index, expert_name)`` for routed experts,
    ``(layer_index, CONV_KEY)`` for the block conv state and
    ``(layer_index, SHARED_KEY)`` for the optional shared expert.
    """

    def detach(self) -> "ExpertStateDict":
        """Detach every carried state from the autograd graph (streaming)."""
        return ExpertStateDict({k: _detach(v) for k, v in self.items()})

    def get_layer(self, layer_idx: int) -> "ExpertStateDict":
        """Slice holding a single layer's entries."""
        return ExpertStateDict({k: v for k, v in self.items() if k[0] == layer_idx})
