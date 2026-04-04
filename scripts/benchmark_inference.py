"""Benchmark: run 100 inference passes and report latency statistics.

Runs 100 inference passes on synthetic (1,4,256,256) float32 frames and reports
mean, p50, p95, and p99 latency in milliseconds. Use this to validate that the
inference latency budget (500ms target; see config/default.toml) is met on the
target hardware (Jetson Xavier AGX/NX).

Usage
-----
    python scripts/benchmark_inference.py [--device cpu|cuda]

Satisfies: §7 of PACT_SW_ARCH.md (scripts/benchmark_inference.py)
"""

from __future__ import annotations

# stdlib
import argparse
import sys

# third-party
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark PACT segmentation model inference latency."
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Torch device to run inference on (default: cpu)",
    )
    parser.add_argument(
        "--n-passes",
        type=int,
        default=100,
        help="Number of inference passes to run (default: 100)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: run N inference passes and print latency statistics."""
    args = parse_args()
    print(f"Benchmarking PACT inference on device='{args.device}' ({args.n_passes} passes)...")
    print("Building randomly initialized model (no weights needed for benchmarking)...")

    # TODO: run N inference passes and collect timing
    # import time
    # import torch
    # from pact.model.architecture import build_model
    #
    # model = build_model(encoder_weights=None).to(args.device).eval()
    # dummy = torch.zeros(1, 4, 256, 256, dtype=torch.float32, device=args.device)
    #
    # # Warm-up passes (excluded from timing)
    # for _ in range(10):
    #     with torch.no_grad():
    #         _ = model(dummy)
    #
    # latencies_ms = []
    # for _ in range(args.n_passes):
    #     if args.device == "cuda":
    #         torch.cuda.synchronize()
    #     t0 = time.perf_counter()
    #     with torch.no_grad():
    #         _ = model(dummy)
    #     if args.device == "cuda":
    #         torch.cuda.synchronize()
    #     t1 = time.perf_counter()
    #     latencies_ms.append((t1 - t0) * 1000.0)
    #
    # latencies = np.array(latencies_ms, dtype=np.float64)
    # print(f"\nResults ({args.n_passes} passes, device={args.device}):")
    # print(f"  Mean:   {latencies.mean():.1f} ms")
    # print(f"  Median (p50): {np.percentile(latencies, 50):.1f} ms")
    # print(f"  p95:    {np.percentile(latencies, 95):.1f} ms")
    # print(f"  p99:    {np.percentile(latencies, 99):.1f} ms")
    # print(f"  Min:    {latencies.min():.1f} ms")
    # print(f"  Max:    {latencies.max():.1f} ms")
    # budget_ms = 500.0
    # n_over = int((latencies > budget_ms).sum())
    # print(f"\n  Latency budget: {budget_ms:.0f} ms")
    # print(f"  Passes over budget: {n_over}/{args.n_passes} ({100*n_over/args.n_passes:.1f}%)")

    print("TODO: implement — run 100 inference passes and report mean/p50/p95/p99 latency.")
    print("      See commented-out implementation above.")
    sys.exit(0)


if __name__ == "__main__":
    main()
