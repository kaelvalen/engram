"""Sequential token-by-token SGMSBlock reference (spec §9 ground truth).

Routes every token individually through the experts' T==1 decode paths.
Ground truth for the equivalence tests — correct by inspection, not on the
hot path.  Learned-mode routing and optional expert knockout only.
"""

from __future__ import annotations

import torch

from .block import SGMSBlock
from .registry import expert_forward
from .router import RoutingOutput
from .state import CONV_KEY, SHARED_KEY, ExpertStateDict


def sequential_block_reference(
    block: SGMSBlock,
    x: torch.Tensor,
    states: ExpertStateDict | None = None,
    exclude: set[str] | None = None,
) -> tuple[torch.Tensor, ExpertStateDict, RoutingOutput]:
    """Reference semantics of SGMSBlock.forward, one token at a time.

    The router is pointwise in t, so dense and sequential routing decisions
    coincide.  Non-selected experts still advance their (masked) state every
    step — decaying (SSD default, D1) or frozen (decay_on_skip=False) —
    exactly as in the dense-masked path.
    """
    cfg = block.cfg
    i = block.layer_idx
    B, T, _ = x.shape
    names = list(block.experts.keys())
    drop = {names.index(n) for n in exclude} if exclude else None

    routing = block.router(x, exclude=drop)

    r = x
    x_n = block.norm1(x)
    conv_in = states.get((i, CONV_KEY)) if states is not None else None
    x_c, conv_new = block.conv(x_n, conv_in)

    updates = ExpertStateDict({(i, CONV_KEY): conv_new})
    y = torch.zeros_like(x)
    for e, (name, expert) in enumerate(block.experts.items()):
        st = states.get((i, name)) if states is not None else None
        for t in range(T):
            m_t = routing.mask[:, t : t + 1, e].contiguous()
            y_t, st = expert_forward(name, expert, x_c[:, t : t + 1], st, m_t, cfg)
            y[:, t] = y[:, t] + routing.gates[:, t, e : e + 1] * y_t[:, 0]
        updates[(i, name)] = st

    if block.shared is not None:
        st = states.get((i, SHARED_KEY)) if states is not None else None
        for t in range(T):
            y_t, st = expert_forward(
                cfg.shared_expert, block.shared, x_c[:, t : t + 1], st, None, cfg
            )
            y[:, t] = y[:, t] + y_t[:, 0]
        updates[(i, SHARED_KEY)] = st

    out = r + y
    out = out + block.ffn(block.norm2(out))
    return out, updates, routing
