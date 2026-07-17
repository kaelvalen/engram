"""float64 gradcheck over the full MoM path (spec §9, R4): router + both
experts, including the masked write path and the stability objectives.
"""

from __future__ import annotations

import torch
from mom.block import MoMBlock
from mom.config import MoMConfig
from mom.losses import mom_auxiliary_loss
from mom.registry import build_expert
from torch.autograd import gradcheck
from torch.func import functional_call


def _cfg(**overrides):
    return MoMConfig(
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        ssd_state_dim=4,
        delta_chunk_size=4,
        scan_backend="reference",
        router_init_std=0.5,  # well-separated logits ⇒ stable top-k under FD eps
        **overrides,
    )


def _block(cfg):
    torch.manual_seed(0)
    return MoMBlock(cfg, layer_idx=0).to(torch.float64).eval()


def test_gradcheck_ssd_expert_masked():
    cfg = _cfg()
    torch.manual_seed(1)
    expert = build_expert("ssd", cfg).to(torch.float64).eval()
    x = torch.randn(1, 6, 8, dtype=torch.float64, requires_grad=True)
    mask = (torch.rand(1, 6) > 0.3).double()

    assert gradcheck(lambda t: expert(t, None, write_mask=mask)[0], (x,), raise_exception=True)


def test_gradcheck_gdr_expert_masked():
    cfg = _cfg()
    torch.manual_seed(1)
    expert = build_expert("gdr", cfg).to(torch.float64).eval()
    x = torch.randn(1, 6, 8, dtype=torch.float64, requires_grad=True)
    mask = (torch.rand(1, 6) > 0.3).double()

    assert gradcheck(
        lambda t: expert(t, None, write_mask=mask, freeze_on_mask=True)[0],
        (x,),
        raise_exception=True,
    )


def test_gradcheck_router_objectives():
    cfg = _cfg()
    block = _block(cfg)
    x = torch.randn(1, 5, 8, dtype=torch.float64)
    w = block.router.weight.clone().requires_grad_(True)

    def fn(weight):
        routing = functional_call(block.router, {"weight": weight}, (x,))
        total, _ = mom_auxiliary_loss([routing], cfg.lambda_bal, cfg.lambda_z)
        return total

    assert gradcheck(fn, (w,), raise_exception=True)


def test_gradcheck_full_block_input_k1():
    block = _block(_cfg(top_k=1))
    x = torch.randn(1, 6, 8, dtype=torch.float64, requires_grad=True)
    assert gradcheck(lambda t: block(t)[0], (x,), raise_exception=True)


def test_gradcheck_full_block_input_k2():
    block = _block(_cfg(top_k=2))
    x = torch.randn(1, 6, 8, dtype=torch.float64, requires_grad=True)
    assert gradcheck(lambda t: block(t)[0], (x,), raise_exception=True)


def test_gradcheck_full_block_router_weight_k2():
    """k=2 gates are non-trivial ⇒ router weight receives gate-path gradient."""
    block = _block(_cfg(top_k=2))
    x = torch.randn(1, 5, 8, dtype=torch.float64)
    w = block.router.weight.clone().requires_grad_(True)

    def fn(weight):
        y, _, routing = functional_call(block, {"router.weight": weight}, (x,))
        aux, _ = mom_auxiliary_loss([routing], 1e-2, 1e-3)
        return torch.cat([y.reshape(-1), aux.reshape(-1)])

    assert gradcheck(fn, (w,), raise_exception=True)


def test_gradcheck_shared_expert_path():
    block = _block(_cfg(shared_expert="ssd"))
    x = torch.randn(1, 6, 8, dtype=torch.float64, requires_grad=True)
    assert gradcheck(lambda t: block(t)[0], (x,), raise_exception=True)


def test_gradcheck_decay_on_skip_false_path():
    block = _block(_cfg(decay_on_skip=False, gdr_decay_on_skip=False))
    x = torch.randn(1, 6, 8, dtype=torch.float64, requires_grad=True)
    assert gradcheck(lambda t: block(t)[0], (x,), raise_exception=True)
