"""Streaming state container (spec §3.6).

Global state = { (ℓ, e) → State }, keyed by layer index and expert name.
Two reserved non-expert keys carry the block's conv state and the optional
always-on shared expert's state.
"""

from __future__ import annotations

import torch

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


def slice_expert_state(state, idx: int | slice | torch.Tensor):
    """Slice an expert state along the batch dimension."""
    if state is None:
        return None
    s_idx = slice(idx, idx + 1) if isinstance(idx, int) else idx
    if isinstance(state, torch.Tensor):
        return state[s_idx]
    if hasattr(state, "S"):  # DeltaState
        from engram.modules.delta import DeltaState

        return DeltaState(S=state.S[s_idx])
    if hasattr(state, "k") and hasattr(state, "v") and hasattr(state, "pos"):  # SWAState
        from engram.modules.attention import SWAState

        return SWAState(k=state.k[s_idx], v=state.v[s_idx], pos=state.pos[s_idx])
    raise TypeError(f"unsupported state type for slicing: {type(state)!r}")


def cat_expert_states(states_list: list):
    """Concatenate a list of batch-sliced expert states along the batch dimension."""
    if not states_list:
        raise ValueError("states_list cannot be empty")
    first = states_list[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(states_list, dim=0)
    if hasattr(first, "S"):  # DeltaState
        from engram.modules.delta import DeltaState

        return DeltaState(S=torch.cat([s.S for s in states_list], dim=0))
    if hasattr(first, "k") and hasattr(first, "v") and hasattr(first, "pos"):  # SWAState
        from engram.modules.attention import SWAState

        return SWAState(
            k=torch.cat([s.k for s in states_list], dim=0),
            v=torch.cat([s.v for s in states_list], dim=0),
            pos=torch.cat([s.pos for s in states_list], dim=0),
        )
    raise TypeError(f"unsupported state type for concatenation: {type(first)!r}")
