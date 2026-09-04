"""Analytic Orin Super latency, duty cycle, and derived FDIR budgets.

Kernel time is ``max(compute / ETA_COMPUTE, memory)``. Wall time is
``kernel / ETA_WRAP``. Search is classifier only. Detect is classifier plus
segmentor. Expected and timeout come from the shipped mixed-knee detect wall
(classifier FP16 plus segmentor INT8).

Contains:
  - BoundLatency / DutyCycle / DerivedBudget.
  - peak_tflops / compute_bound_ms / memory_bound_ms / model_latency.
  - duty_cycle / derive_budget / mean_time_ms.
  - optional ORT bench CSV (not plotted as Orin).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from analysis.studies.orin_nano_full_frame_inference.assumptions import (
    BENCH_CSV,
    BW_GBPS,
    CLS_ONNX,
    DATA_DIR,
    ETA_COMPUTE,
    ETA_WRAP,
    FLIGHT_PRECISION,
    FP16_TFLOPS,
    FP32_TFLOPS,
    FRAME_H_PX,
    FRAME_W_PX,
    IN_CHANNELS,
    INT8_DENSE_TOPS,
    PREFERRED_ORT_PROVIDERS,
    SEG_ONNX,
    TIMEOUT_MULT,
)
from analysis.studies.orin_nano_full_frame_inference.cost import (
    bytes_moved,
    cls_flops_ff_g,
    seg_flops_ff_g,
)

PRECISIONS: tuple[str, ...] = ("fp32", "fp16", "int8")


@dataclass(frozen=True, slots=True)
class BoundLatency:
    """Compute-bound and memory-bound times for one forward pass.

    Attributes:
        compute_ms: Peak-FLOP time before small-net efficiency.
        memory_ms: DRAM time at 102 GB/s.
        kernel_ms: ``max(compute / eta, memory)``.
        wall_ms: Kernel time divided by wrap efficiency.
    """

    compute_ms: float
    memory_ms: float
    kernel_ms: float
    wall_ms: float


@dataclass(frozen=True, slots=True)
class DutyCycle:
    """Search versus detect wall times at one precision.

    Attributes:
        t_search_ms: Classifier wall time.
        t_detect_ms: Classifier plus segmentor wall time.
    """

    t_search_ms: float
    t_detect_ms: float


@dataclass(frozen=True, slots=True)
class DerivedBudget:
    """Flight expected latency and FDIR timeout derived from detect wall.

    Attributes:
        expected_ms: ``ceil(t_detect)`` for the shipped precision.
        timeout_ms: ``ceil(TIMEOUT_MULT * expected_ms)``.
        t_detect_ms: Detect wall used as the input.
        precision: Precision the budget was derived from.
    """

    expected_ms: int
    timeout_ms: int
    t_detect_ms: float
    precision: str


def peak_tflops(precision: str) -> float:
    """Return the Super peak used as the compute bound.

    Args:
        precision: ``fp32``, ``fp16``, or ``int8``.

    Returns:
        TFLOPS (INT8 uses dense TOPS as the equivalent).

    Raises:
        ValueError: If ``precision`` is unknown.
    """
    if precision == "fp32":
        return FP32_TFLOPS
    if precision == "fp16":
        return FP16_TFLOPS
    if precision == "int8":
        return INT8_DENSE_TOPS
    raise ValueError(f"unknown precision {precision!r}")


def compute_bound_ms(flops_g: float, precision: str) -> float:
    """Return peak compute time in milliseconds.

    ``time_ms = GFLOP / TFLOPS`` because ``1 TFLOP/s = 1000 GFLOP/s`` and
    ``1000 * GFLOP / (TFLOPS * 1000) = GFLOP / TFLOPS``.

    Args:
        flops_g: One-pass G count.
        precision: Named precision.

    Returns:
        Milliseconds at 100% of the Super peak.
    """
    return flops_g / peak_tflops(precision)


def memory_bound_ms(nbytes: float) -> float:
    """Return DRAM time in milliseconds at Super bandwidth.

    Args:
        nbytes: Bytes moved.

    Returns:
        Milliseconds at ``BW_GBPS``.
    """
    return 1000.0 * nbytes / (BW_GBPS * 1e9)


def model_latency(flops_g: float, nbytes: float, precision: str) -> BoundLatency:
    """Return kernel and wall time for one forward pass.

    Args:
        flops_g: One-pass G count.
        nbytes: Bytes moved.
        precision: Named precision.

    Returns:
        BoundLatency with compute, memory, kernel, and wall.
    """
    compute = compute_bound_ms(flops_g, precision)
    memory = memory_bound_ms(nbytes)
    kernel = max(compute / ETA_COMPUTE, memory)
    wall = kernel / ETA_WRAP
    return BoundLatency(compute_ms=compute, memory_ms=memory, kernel_ms=kernel, wall_ms=wall)


def cls_latency(precision: str) -> BoundLatency:
    """Return classifier latency at ``precision``."""
    return model_latency(cls_flops_ff_g(), bytes_moved("cls", precision), precision)


def seg_latency(precision: str) -> BoundLatency:
    """Return segmentor latency at ``precision``."""
    return model_latency(seg_flops_ff_g(), bytes_moved("seg", precision), precision)


def duty_cycle(precision: str) -> DutyCycle:
    """Return search (cls) and detect (cls+seg) wall times.

    Args:
        precision: Named precision applied to both nets.

    Returns:
        DutyCycle with ``t_search = t_cls`` and ``t_detect = t_cls + t_seg``.
    """
    t_cls = cls_latency(precision).wall_ms
    t_seg = seg_latency(precision).wall_ms
    return DutyCycle(t_search_ms=t_cls, t_detect_ms=t_cls + t_seg)


def derive_budget(precision: str = FLIGHT_PRECISION) -> DerivedBudget:
    """Ceil detect wall into expected latency and FDIR timeout.

    Args:
        precision: Named precision for both nets, or ``mixed`` for the
            shipped knee (classifier FP16, segmentor INT8).

    Returns:
        DerivedBudget with integer millisecond fields.
    """
    if precision == "mixed":
        detect = mixed_knee_detect_ms()
    else:
        detect = duty_cycle(precision).t_detect_ms
    expected = int(math.ceil(detect))
    timeout = int(math.ceil(TIMEOUT_MULT * expected))
    return DerivedBudget(
        expected_ms=expected,
        timeout_ms=timeout,
        t_detect_ms=detect,
        precision=precision,
    )


def mean_time_ms(precision: str, positive_rate: float) -> float:
    """Return expected wall time at a plume-positive rate.

    Args:
        precision: Named precision.
        positive_rate: Fraction of frames that run the segmentor, in ``[0, 1]``.

    Returns:
        ``t_cls + p * t_seg``.
    """
    t_cls = cls_latency(precision).wall_ms
    t_seg = seg_latency(precision).wall_ms
    return t_cls + positive_rate * t_seg


def mixed_knee_detect_ms() -> float:
    """Return detect wall for the quality knee: classifier FP16, segmentor INT8."""
    return cls_latency("fp16").wall_ms + seg_latency("int8").wall_ms


def run_ort_bench(csv_path: Path | None = None) -> Path | None:
    """Time factory ONNX with available ORT providers. CPU is not Orin.

    Args:
        csv_path: Destination CSV. Default is ``DATA_DIR / ort_bench.csv``.

    Returns:
        Written path, or None when onnxruntime or the artifacts are missing.
    """
    try:
        import numpy as np
        import onnxruntime
    except ImportError:
        return None
    dest = csv_path if csv_path is not None else BENCH_CSV
    available = set(onnxruntime.get_available_providers())
    providers = [name for name in PREFERRED_ORT_PROVIDERS if name in available]
    if not providers:
        return None
    dummy = np.zeros((1, IN_CHANNELS, FRAME_H_PX, FRAME_W_PX), dtype=np.float32)
    rows: list[dict[str, str]] = []
    warmup = 3
    repeats = 10
    for label, path in (("classifier", CLS_ONNX), ("segmentor", SEG_ONNX)):
        artifact = Path(path)
        if not artifact.is_file():
            continue
        session = onnxruntime.InferenceSession(str(artifact), providers=providers)
        feed = {session.get_inputs()[0].name: dummy}
        for _ in range(warmup):
            session.run(None, feed)
        import time as time_mod

        samples: list[float] = []
        for _ in range(repeats):
            t0 = time_mod.perf_counter()
            session.run(None, feed)
            samples.append(1000.0 * (time_mod.perf_counter() - t0))
        host = "laptop_gpu" if "CUDAExecutionProvider" in session.get_providers() else "cpu"
        rows.append(
            {
                "model": label,
                "host": host,
                "providers": ",".join(session.get_providers()),
                "median_ms": f"{sorted(samples)[len(samples) // 2]:.4f}",
                "p95_ms": f"{sorted(samples)[int(0.95 * (len(samples) - 1))]:.4f}",
            }
        )
    if not rows:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["model", "host", "providers", "median_ms", "p95_ms"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return dest


def load_bench_csv(path: Path | None = None) -> list[dict[str, str]]:
    """Load an optional ORT bench CSV.

    Args:
        path: CSV path. Default is ``BENCH_CSV``.

    Returns:
        Row dicts, or an empty list when the file is missing.
    """
    dest = path if path is not None else BENCH_CSV
    if not dest.is_file():
        return []
    with dest.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_data_dir() -> Path:
    """Create the gitignored data directory if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
