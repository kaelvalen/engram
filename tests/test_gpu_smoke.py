"""CUDA smoke tests for GPU-accelerated backends, memory bounding, and gathered routing.

These tests require a CUDA-capable device and are marked with ``@pytest.mark.gpu``.
They verify:
1. Basic CUDA tensor allocation & forward pass through SGMSBlock.
2. FLA Triton kernel execution for GatedDeltaRule on CUDA, matching reference.
3. SSD chunked linear recurrence bounds peak activation memory on long sequences (T=4096).
4. Multi-chunk gathered state streaming on CUDA for B > 1.
"""

from __future__ import annotations

import pytest
import torch
from engram.modules.delta import GatedDeltaRule, _load_fla
from engram.modules.ssd import SSDMixer
from sgms import SGMSBlock, SGMSConfig

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available"),
]


@pytest.fixture(autouse=True)
def _cuda_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_cuda_sgms_block_forward_dense_and_gathered():
    device = "cuda:0"
    B, T, D = 2, 64, 128
    x = torch.randn(B, T, D, device=device)

    for mode in ("dense_masked", "gathered"):
        cfg = SGMSConfig(
            hidden_dim=D,
            num_heads=4,
            ssd_state_dim=32,
            experts=("ssd", "gdr"),
            top_k=1,
            execution_mode=mode,
            decay_on_skip=False,
            gdr_decay_on_skip=False,
        )
        block = SGMSBlock(cfg, layer_idx=0).to(device)
        y, updates, routing = block(x)

        assert y.device.type == "cuda"
        assert y.shape == (B, T, D)
        assert torch.isfinite(y).all()
        assert routing.gates.device.type == "cuda"
        assert routing.mask.device.type == "cuda"


def test_cuda_gdr_fla_backend_equivalence():
    if _load_fla() is None:
        pytest.skip("FLA (flash-linear-attention) is not installed")

    device = "cuda:0"
    B, T, D, H = 2, 128, 128, 4
    torch.manual_seed(42)
    # FLA kernel operates on bfloat16 / float16
    x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)

    # Instantiate reference and fla models sharing identical weights
    torch.manual_seed(123)
    model_ref = (
        GatedDeltaRule(hidden_dim=D, num_heads=H, backend="reference")
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    torch.manual_seed(123)
    model_fla = (
        GatedDeltaRule(hidden_dim=D, num_heads=H, backend="fla")
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )

    with torch.no_grad():
        out_ref, state_ref = model_ref(x)
        out_fla, state_fla = model_fla(x)

    # FLA chunked kernel vs reference chunked kernel numerical tolerance
    assert out_fla.shape == (B, T, D)
    assert out_ref.shape == (B, T, D)
    torch.testing.assert_close(out_fla, out_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(state_fla.S, state_ref.S, rtol=1e-2, atol=1e-2)


def test_cuda_ssd_chunking_bounded_memory():
    device = "cuda:0"
    B, T, D, H, N = 2, 4096, 256, 8, 64
    x = torch.randn(B, T, D, device=device, dtype=torch.float32)

    model = SSDMixer(hidden_dim=D, num_heads=H, state_dim=N, chunk_size=256).to(device).eval()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_allocated = torch.cuda.memory_allocated()

    with torch.inference_mode():
        y, _ = model(x)
        torch.cuda.synchronize()

    peak_allocated = torch.cuda.max_memory_allocated()
    delta_peak_mb = (peak_allocated - base_allocated) / (1024 * 1024)

    assert y.shape == (B, T, D)
    assert torch.isfinite(y).all()
    # Unchunked scan at T=4096 would allocate multiple gigabytes of dBx and scan states.
    # Chunked recurrence (chunk_size=256) bounds activation memory to well below 500 MB.
    assert delta_peak_mb < 500, f"Expected chunked peak memory < 500 MB, got {delta_peak_mb:.1f} MB"


def test_cuda_gathered_streaming_handoff():
    device = "cuda:0"
    B, T, D = 2, 128, 128
    chunk_len = 64
    x = torch.randn(B, T, D, device=device)

    cfg = SGMSConfig(
        hidden_dim=D,
        num_heads=4,
        ssd_state_dim=32,
        experts=("ssd", "gdr"),
        top_k=1,
        execution_mode="gathered",
        decay_on_skip=False,
        gdr_decay_on_skip=False,
    )
    block = SGMSBlock(cfg, layer_idx=0).to(device).eval()

    # Full sequence forward
    with torch.inference_mode():
        y_full, _, _ = block(x)

    # Chunked streaming forward
    y_chunks = []
    states = None
    with torch.inference_mode():
        for start in range(0, T, chunk_len):
            x_c = x[:, start : start + chunk_len]
            y_c, states, _ = block(x_c, states=states)
            y_chunks.append(y_c)

    y_streamed = torch.cat(y_chunks, dim=1)
    torch.testing.assert_close(y_streamed, y_full, rtol=1e-4, atol=1e-4)
