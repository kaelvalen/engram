"""Conditional compute wall-clock and peak memory benchmark suite.

Measures throughput (tokens/sec), step latency (ms), and peak VRAM (MB) across
varying sequence lengths for:
1. Dense Masked SGMS (SSD + GDR, wall-clock decay)
2. Gathered Sparse SGMS (SSD + GDR, event-time sparse dispatch)
3. Pure Dense SSD baseline
4. Pure Dense GDR baseline

Usage:
    python scripts/benchmark_conditional_compute.py --device cpu --seq-lens 64,128 --batch-size 2 --runs 5
    python scripts/benchmark_conditional_compute.py --device cuda --seq-lens 256,1024,4096 --batch-size 4 --runs 20
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# NixOS / driver linker path setup
_nixos_cuda_lib = Path("/run/opengl-driver/lib")
if (_nixos_cuda_lib / "libcuda.so.1").exists():
    os.environ.setdefault("TRITON_LIBCUDA_PATH", str(_nixos_cuda_lib))

import torch  # noqa: E402
from engram.modules.delta import GatedDeltaRule  # noqa: E402
from engram.modules.ssd import SSDMixer  # noqa: E402
from sgms import SGMSBlock, SGMSConfig  # noqa: E402


def build_models(
    dim: int, heads: int, state_dim: int, top_k: int, device: str, dtype: torch.dtype = torch.float32
) -> dict[str, torch.nn.Module]:
    """Instantiate the 4 comparison models."""
    cfg_dense = SGMSConfig(
        hidden_dim=dim,
        num_heads=heads,
        ssd_state_dim=state_dim,
        experts=("ssd", "gdr"),
        top_k=top_k,
        execution_mode="dense_masked",
        decay_on_skip=False,
        gdr_decay_on_skip=False,
    )
    dense_sgms = SGMSBlock(cfg_dense, layer_idx=0).to(device=device, dtype=dtype).eval()

    cfg_gathered = SGMSConfig(
        hidden_dim=dim,
        num_heads=heads,
        ssd_state_dim=state_dim,
        experts=("ssd", "gdr"),
        top_k=top_k,
        execution_mode="gathered",
        decay_on_skip=False,
        gdr_decay_on_skip=False,
    )
    gathered_sgms = SGMSBlock(cfg_gathered, layer_idx=0).to(device=device, dtype=dtype).eval()

    pure_ssd = (
        SSDMixer(hidden_dim=dim, num_heads=heads, state_dim=state_dim)
        .to(device=device, dtype=dtype)
        .eval()
    )
    pure_gdr = (
        GatedDeltaRule(hidden_dim=dim, num_heads=heads, backend="auto")
        .to(device=device, dtype=dtype)
        .eval()
    )

    return {
        "Dense Masked SGMS": dense_sgms,
        "Gathered Sparse SGMS": gathered_sgms,
        "Dense Pure SSD": pure_ssd,
        "Dense Pure GDR": pure_gdr,
    }


def benchmark_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    runs: int,
    warmup: int,
    is_cuda: bool,
    device: str,
) -> tuple[float, float, float | None]:
    """Runs timed benchmark for a single model and input tensor.

    Returns:
        (latency_ms, throughput_tok_s, peak_vram_mb)
    """
    with torch.inference_mode():
        # Warmup passes
        for _ in range(warmup):
            model(x)

        if is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)

        if is_cuda:
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            peak_bytes = torch.cuda.max_memory_allocated(device)
            peak_vram_mb = peak_bytes / (1024 * 1024)
        else:
            t1 = time.perf_counter()
            peak_vram_mb = None

    elapsed = t1 - t0
    latency_ms = (elapsed / runs) * 1000.0
    b_sz, t_len, _ = x.shape
    throughput = (b_sz * t_len * runs) / elapsed
    return latency_ms, throughput, peak_vram_mb


def run_benchmarks(args: argparse.Namespace) -> dict:
    device = args.device
    is_cuda = torch.device(device).type == "cuda"
    seq_lens = [int(s.strip()) for s in args.seq_lens.split(",") if s.strip()]
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    print(f"=== SGMS Conditional Compute Benchmark ===")
    print(f"Device: {device} | PyTorch: {torch.__version__} | Dtype: {args.dtype}")
    print(f"Batch: {args.batch_size} | Dim: {args.dim} | Heads: {args.heads} | State Dim: {args.state_dim}")
    print(f"Sequences: {seq_lens} | Top-k: {args.top_k} | Warmup: {args.warmup} | Runs: {args.runs}\n")

    models = build_models(args.dim, args.heads, args.state_dim, args.top_k, device, dtype=dtype)
    all_results = {}

    for t_len in seq_lens:
        print(f"--- Sequence Length T = {t_len} ---")
        x = torch.randn(args.batch_size, t_len, args.dim, device=device, dtype=dtype)
        t_results = {}

        for name, model in models.items():
            latency_ms, throughput, peak_mb = benchmark_model(
                model, x, args.runs, args.warmup, is_cuda, device
            )
            vram_str = f"{peak_mb:8.1f} MB" if peak_mb is not None else "     N/A"
            print(
                f"  {name:<22}: {latency_ms:8.2f} ms | {throughput:>12,.0f} tok/s | Peak VRAM: {vram_str}"
            )
            t_results[name] = {
                "latency_ms": latency_ms,
                "throughput_tok_s": throughput,
                "peak_vram_mb": peak_mb,
            }

        dense_lat = t_results["Dense Masked SGMS"]["latency_ms"]
        gath_lat = t_results["Gathered Sparse SGMS"]["latency_ms"]
        speedup = dense_lat / gath_lat if gath_lat > 0 else 1.0
        print(f"  --> Gathered vs Dense Speedup: {speedup:0.2f}x\n")

        all_results[f"T_{t_len}"] = t_results

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output_json}")

    return all_results


def main():
    p = argparse.ArgumentParser(description="Benchmark conditional compute wall-clock performance and peak memory.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda)")
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32",
        choices=["float32", "bfloat16", "float16"],
        help="Data type for activations and parameters",
    )
    p.add_argument("--batch-size", type=int, default=4, help="Batch size B")
    p.add_argument("--seq-lens", type=str, default="256,1024,4096", help="Comma-separated sequence lengths T")
    p.add_argument("--dim", type=int, default=256, help="Hidden dimension D")
    p.add_argument("--heads", type=int, default=8, help="Number of heads")
    p.add_argument("--state-dim", type=int, default=64, help="State dimension N")
    p.add_argument("--top-k", type=int, default=1, help="Top-k active experts per token")
    p.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    p.add_argument("--runs", type=int, default=10, help="Benchmark timing iterations")
    p.add_argument("--output-json", type=str, default=None, help="Optional output JSON path")
    args = p.parse_args()

    run_benchmarks(args)


if __name__ == "__main__":
    main()
