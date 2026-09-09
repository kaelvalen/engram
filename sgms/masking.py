"""Routing-mask algebra and output combination (spec §3.4, §3.5).

m_{t,e} = 1[e ∈ S_t]                     (write-side mask per expert)
y_t     = Σ_{e ∈ S_t} g_{t,e} · y_{t,e}  (gated combination)
"""

from __future__ import annotations

import torch


def topk_mask(indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Float {0,1} mask [B, T, K] from top-k expert indices [B, T, k].

    Idempotent by construction (entries are exactly 0 or 1).
    """
    if indices.dtype != torch.long:
        raise TypeError(f"indices must be long, got {indices.dtype}")
    B, T, _ = indices.shape
    mask = torch.zeros(B, T, num_experts, dtype=torch.float32, device=indices.device)
    mask.scatter_(-1, indices, 1.0)
    return mask


def combine_expert_outputs(outputs: list[torch.Tensor], gates: torch.Tensor) -> torch.Tensor:
    """y_t = Σ_e g_{t,e} · y_{t,e}  (gates are 0 outside each token's S_t).

    outputs: list of K tensors [B, T, D]; gates: [B, T, K].
    """
    if len(outputs) != gates.shape[-1]:
        raise ValueError(f"{len(outputs)} outputs but gates have K={gates.shape[-1]}")
    if not outputs:
        raise ValueError("at least one expert output is required")
    # Avoid materializing a full [B, T, K, D] stacked tensor across experts.
    # Accumulate directly along the hidden dimension to save peak memory.
    out = outputs[0] * gates[..., 0:1]
    for e in range(1, len(outputs)):
        out = out.addcmul(outputs[e], gates[..., e : e + 1])
    return out
