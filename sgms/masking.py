"""Routing-mask algebra and output combination (spec §3.4, §3.5).

m_{t,e} = 1[e ∈ S_t]                     (write-side mask per expert)
y_t     = Σ_{e ∈ S_t} g_{t,e} · y_{t,e}  (gated combination)
"""

from __future__ import annotations

import torch


def topk_mask(
    indices: torch.Tensor, num_experts: int, dtype: torch.dtype = torch.bool
) -> torch.Tensor:
    """Mask [B, T, K] from top-k expert indices [B, T, k].

    Defaults to torch.bool for a 4x smaller memory footprint than float32.
    """
    if indices.dtype != torch.long:
        raise TypeError(f"indices must be long, got {indices.dtype}")
    B, T, _ = indices.shape
    mask = torch.zeros(B, T, num_experts, dtype=dtype, device=indices.device)
    val = True if dtype == torch.bool else 1.0
    mask.scatter_(-1, indices, val)
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
