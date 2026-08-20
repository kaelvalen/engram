"""Throughput micro-benchmark for the paper's throughput plot.

Measures tokens/sec for the SSD scan (associative_scan vs reference vs compiled)
and the gated delta rule (reference vs FLA Triton, if installed) across a few
state/head dims. Runs on CPU (smoke) or CUDA (real numbers). The manuscript
should report CUDA numbers from the user's own hardware, not these CPU values.

    python scripts/bench_throughput.py --device cuda --seq-len 4096
"""

from __future__ import annotations

import argparse
import time

import torch
from engram.modules.delta import GatedDeltaRule, _load_fla
from engram.modules.ssd import SSDMixer


def _timed(fn, iters: int, warmup: int, device: str) -> float:
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bench_ssd(device, B, T, D, H, N, iters, warmup):
    print(f"\n== SSD scan  [B={B} T={T} D={D} H={H} N={N}] ==")
    x = torch.randn(B, T, D, device=device)
    for backend in ("reference", "assoc"):
        m = SSDMixer(D, H, N, scan_backend=backend).to(device).eval()
        with torch.no_grad():
            dt = _timed(lambda: m(x), iters, warmup, device)
        print(f"  {backend:10s}: {B * T / dt:>12,.0f} tok/s  ({dt * 1e3:.2f} ms)")
    # torch.compile path (GPU mainly)
    try:
        m = SSDMixer(D, H, N, scan_backend="assoc").to(device).eval()
        mc = torch.compile(m)
        with torch.no_grad():
            dt = _timed(lambda: mc(x), iters, warmup, device)
        print(f"  {'compiled':10s}: {B * T / dt:>12,.0f} tok/s  ({dt * 1e3:.2f} ms)")
    except Exception as e:
        print(f"  compiled  : skipped ({e})")


def bench_delta(device, B, T, D, H, iters, warmup):
    print(f"\n== Gated delta  [B={B} T={T} D={D} H={H}] ==")
    x = torch.randn(B, T, D, device=device)
    m = GatedDeltaRule(D, H, backend="reference").to(device).eval()
    with torch.no_grad():
        dt = _timed(lambda: m(x), iters, warmup, device)
    print(f"  {'reference':10s}: {B * T / dt:>12,.0f} tok/s  ({dt * 1e3:.2f} ms)")
    if _load_fla() is not None and device == "cuda":
        mf = GatedDeltaRule(D, H, backend="fla").to(device).eval()
        with torch.no_grad():
            dt = _timed(lambda: mf(x), iters, warmup, device)
        print(f"  {'fla':10s}: {B * T / dt:>12,.0f} tok/s  ({dt * 1e3:.2f} ms)")
    else:
        print("  fla       : skipped (FLA or CUDA unavailable)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    print(f"device={args.device}  torch={torch.__version__}")
    for N in (16, 64, 128):  # sweep state dim / head dim regimes
        bench_ssd(
            args.device, args.batch, args.seq_len, args.dim, args.heads, N, args.iters, args.warmup
        )
    bench_delta(
        args.device, args.batch, args.seq_len, args.dim, args.heads, args.iters, args.warmup
    )


if __name__ == "__main__":
    main()
