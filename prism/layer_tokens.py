from __future__ import annotations

# Valid per-layer role tokens for PRISM backbones.
#   "s4"    — SSM sequence-mixer slot. Concrete implementation is chosen by
#             `ssm_kind` ("ssd" → Mamba-2-style selective scan,
#             "s4d_legacy" → S4D-Complex).
#   "delta" — Gated Delta Rule (matrix-valued associative memory).
#   "swa"   — Sliding-window attention (for H1-style hybrid ablations).
#
# Kept in a dependency-free module so ``prism.config`` can validate patterns
# without importing the full modules package (and its PyTorch dependency).
LAYER_TOKENS = ("s4", "delta", "swa")
