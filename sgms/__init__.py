"""SGMS — Mixture of Memory Primitives.

Replaces the fixed hybrid block at each ENGRAM layer with a bank of
heterogeneous memory primitives (SSD, GDR; SWA in v2) and a per-token
router.  See SGMS-Architecture-Spec.md; ENGRAM mixers are consumed through
sgms.registry, never modified beyond additive write-mask flags.
"""

from .block import SGMSBlock
from .config import SGMSConfig
from .losses import load_balancing_loss, router_z_loss, sgms_auxiliary_loss
from .masking import combine_expert_outputs, topk_mask
from .model import SGMSLM
from .registry import EXPERT_NAMES, build_expert, expert_forward, expert_param_count
from .router import RoutingOutput, TokenRouter, routing_stats
from .state import CONV_KEY, SHARED_KEY, ExpertStateDict

__all__ = [
    "CONV_KEY",
    "EXPERT_NAMES",
    "ExpertStateDict",
    "SGMSBlock",
    "SGMSConfig",
    "SGMSLM",
    "RoutingOutput",
    "SHARED_KEY",
    "TokenRouter",
    "build_expert",
    "combine_expert_outputs",
    "expert_forward",
    "expert_param_count",
    "load_balancing_loss",
    "sgms_auxiliary_loss",
    "router_z_loss",
    "routing_stats",
    "topk_mask",
]
