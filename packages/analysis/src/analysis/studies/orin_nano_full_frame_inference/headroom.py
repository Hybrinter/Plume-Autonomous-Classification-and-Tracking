"""Headroom and max-upload size for the Orin Nano Super study.

The shipped mixed knee sits on the activation-memory floor of the 1024x1224
float32 band plane. Peak FLOPs and 8 GB DRAM therefore look idle, while the
4 ms expected budget is tight by construction (ceil of detect wall). Jetson
memory is unified LPDDR5, not discrete VRAM. AGX Orin uses the same
Cortex-A78AE CPU with more cores and clock; it is not a different CPU.

Contains:
  - Utilization / SizeCeiling / CatalogFit / TensorWorkingSet / BoardCompare.
  - utilization / memory_floor_wall_ms / max_params_for_wall.
  - factory_scaled_pair_bytes / size_ceilings / catalog_fits.
  - tensor_working_set / board_compares.
"""

from __future__ import annotations

from dataclasses import dataclass

from analysis.studies.orin_nano_full_frame_inference.assumptions import (
    BW_GBPS,
    CAMERA_PREPROCESS_GB,
    CATALOG,
    CLS_FLIGHT_ONNX_BYTES,
    CLS_FLIGHT_PRECISION,
    COMPARE_BOARDS,
    DISK_MODEL_COPIES,
    DRAM_GB,
    ETA_COMPUTE,
    ETA_WRAP,
    GPU_SESSIONS_DURING_ACTIVATE,
    MAX_DAILY_UPLINK_BYTES,
    MAX_FRAME_RATE_HZ,
    MAX_STORAGE_BYTES,
    MAX_UPLINK_BPS,
    NANO_SUPER,
    ORT_WORKSPACE_GB,
    OS_RUNTIME_GB,
    PAYLOAD_BUS_W,
    REASSEMBLY_RAM_COPIES,
    SEG_FLIGHT_ONNX_BYTES,
    SEG_FLIGHT_PRECISION,
    TRT_ENGINE_OVERHEAD,
    BoardSpec,
    CatalogArch,
    catalog_area_scale,
)
from analysis.studies.orin_nano_full_frame_inference.cost import (
    activation_bytes,
    bytes_moved_for,
    cls_output_bytes,
    flop_per_param,
    input_bytes,
    params_bytes,
    seg_output_bytes,
    weight_bytes,
)
from analysis.studies.orin_nano_full_frame_inference.latency import (
    cls_latency,
    derive_budget,
    mixed_knee_detect_ms,
    model_latency,
    peak_tflops,
    seg_latency,
)


@dataclass(frozen=True, slots=True)
class Utilization:
    """Shipped mixed-knee occupancy against time, compute, and DRAM.

    Attributes:
        detect_wall_ms: Classifier FP16 plus segmentor INT8 wall.
        expected_ms: Ceil detect wall (live ``latency_budget_ms``).
        timeout_ms: Ceil of 5x expected (live ``inference_timeout_ms``).
        frame_period_ms: ``1000 / MAX_FRAME_RATE_HZ``.
        frac_expected: Detect wall over expected.
        frac_timeout: Detect wall over FDIR timeout.
        frac_frame: Detect wall over the 35 Hz frame period.
        kernel_ms: Sum of mixed-knee kernels (cls FP16 + seg INT8).
        gpu_busy_frac_35hz: Kernel time over the 35 Hz period (every frame detect).
        pair_onnx_bytes: Shipped factory ONNX pair on disk.
        io_activation_bytes: Float32 input plus locked live maps.
        dram_free_gb: 8 GB minus named OS / workspace / camera reservations.
    """

    detect_wall_ms: float
    expected_ms: int
    timeout_ms: int
    frame_period_ms: float
    frac_expected: float
    frac_timeout: float
    frac_frame: float
    kernel_ms: float
    gpu_busy_frac_35hz: float
    pair_onnx_bytes: float
    io_activation_bytes: float
    dram_free_gb: float


@dataclass(frozen=True, slots=True)
class SizeCeiling:
    """One independent cap on a classifier+segmentor pair artifact.

    Attributes:
        name: Short label for tables and figures.
        max_pair_bytes: Ceiling in bytes. 0 means the budget cannot cover I/O.
        binds: ``latency``, ``uplink``, ``dram``, or ``storage``.
    """

    name: str
    max_pair_bytes: float
    binds: str


@dataclass(frozen=True, slots=True)
class CatalogFit:
    """One catalog architecture swapped in beside the shipped other net.

    Attributes:
        arch: Catalog row.
        precision: FP16 for classifiers, INT8 for segmentors.
        flops_ff_g: Catalog 128-px G count times 76.5.
        wall_ms: Analytic wall of this net alone.
        pair_detect_ms: This net plus the shipped partner.
        onnx_bytes: ``params * elem`` (no QDQ overhead).
        fits_expected: Pair detect <= expected ms.
        fits_timeout: Pair detect <= FDIR timeout.
        fits_frame: Pair detect <= 35 Hz period.
    """

    arch: CatalogArch
    precision: str
    flops_ff_g: float
    wall_ms: float
    pair_detect_ms: float
    onnx_bytes: float
    fits_expected: bool
    fits_timeout: bool
    fits_frame: bool


def frame_period_ms() -> float:
    """Return milliseconds per frame at the camera max rate."""
    return 1000.0 / MAX_FRAME_RATE_HZ


def dram_free_gb() -> float:
    """Return LPDDR5 left after the named reservation policy.

    Returns:
        Gigabytes. This is a reservation, not a Nano measurement.
    """
    return DRAM_GB - OS_RUNTIME_GB - ORT_WORKSPACE_GB - CAMERA_PREPROCESS_GB


def pair_onnx_bytes() -> float:
    """Return shipped factory ONNX bytes (classifier plus segmentor)."""
    return float(CLS_FLIGHT_ONNX_BYTES + SEG_FLIGHT_ONNX_BYTES)


def utilization() -> Utilization:
    """Return shipped mixed-knee occupancy against the live budgets.

    Returns:
        Utilization with time fractions, GPU busy fraction, and DRAM free.
    """
    budget = derive_budget()
    detect = budget.t_detect_ms
    period = frame_period_ms()
    kernel = cls_latency("fp16").kernel_ms + seg_latency("int8").kernel_ms
    return Utilization(
        detect_wall_ms=detect,
        expected_ms=budget.expected_ms,
        timeout_ms=budget.timeout_ms,
        frame_period_ms=period,
        frac_expected=detect / float(budget.expected_ms),
        frac_timeout=detect / float(budget.timeout_ms),
        frac_frame=detect / period,
        kernel_ms=kernel,
        gpu_busy_frac_35hz=(kernel / period),
        pair_onnx_bytes=pair_onnx_bytes(),
        io_activation_bytes=input_bytes() + activation_bytes(),
        dram_free_gb=dram_free_gb(),
    )


def memory_floor_wall_ms() -> float:
    """Return wall time of a weightless forward pass (I/O plus live maps).

    Returns:
        Milliseconds. The shipped nets sit just above this floor.
    """
    nbytes = input_bytes() + activation_bytes()
    return model_latency(0.0, nbytes, "fp16").wall_ms


def max_params_for_wall(wall_ms: float, precision: str, kind: str) -> float:
    """Return max parameters whose factory-intensity wall is ``wall_ms``.

    Compute-limited params use ``GFLOP = wall * wrap * eta * peak``.
    Memory-limited params use leftover DRAM traffic after I/O and live maps.
    The result is the tighter of the two. Activations stay at the locked
    64-channel estimate, so wider nets are optimistic.

    Args:
        wall_ms: Allowed wall time for this net alone.
        precision: ``fp32``, ``fp16``, or ``int8``.
        kind: ``cls`` or ``seg`` (selects factory FLOP/parameter).

    Returns:
        Parameter count, or 0 when ``wall_ms`` cannot cover the memory floor.
    """
    if wall_ms <= 0.0:
        return 0.0
    intensity = flop_per_param(kind)
    peak = peak_tflops(precision)
    g_compute = wall_ms * ETA_WRAP * ETA_COMPUTE * peak
    params_compute = g_compute * 1e9 / intensity
    kernel_ms = wall_ms * ETA_WRAP
    nbytes = kernel_ms * BW_GBPS * 1e9 / 1000.0
    leftover = nbytes - input_bytes() - activation_bytes()
    if leftover <= 0.0:
        return 0.0
    params_memory = leftover / float(params_bytes(1.0, precision))
    return min(params_compute, params_memory)


def factory_scale_for_detect(budget_ms: float) -> float:
    """Return the linear scale that grows the mixed knee to ``budget_ms``.

    Args:
        budget_ms: Target detect wall.

    Returns:
        ``budget / shipped_detect``. Both nets scale together. Valid while
        weights stay small versus the 80 MB activation traffic.
    """
    detect = mixed_knee_detect_ms()
    return budget_ms / detect


def factory_scaled_pair_bytes(budget_ms: float) -> float:
    """Return factory-pair ONNX bytes scaled to a detect-wall budget.

    Args:
        budget_ms: Target detect wall (expected, timeout, or frame period).

    Returns:
        ``scale * shipped pair bytes``.
    """
    return factory_scale_for_detect(budget_ms) * pair_onnx_bytes()


def uplink_seconds(nbytes: float) -> float:
    """Return seconds to push ``nbytes`` at the configured uplink rate.

    Args:
        nbytes: Artifact bytes.

    Returns:
        ``8 * nbytes / max_uplink_bps``.
    """
    return 8.0 * nbytes / float(MAX_UPLINK_BPS)


def size_ceilings() -> tuple[SizeCeiling, ...]:
    """Return independent pair-size caps, smallest usable first.

    Returns:
        Ceilings for 4 ms expected, 20 ms timeout, 35 Hz frame, daily
        uplink, DRAM during activate, and storage after three copies.
    """
    budget = derive_budget()
    period = frame_period_ms()
    dram_bytes = dram_free_gb() * 1e9
    dram_onnx = dram_bytes / (TRT_ENGINE_OVERHEAD * GPU_SESSIONS_DURING_ACTIVATE)
    reassembly = dram_bytes / float(REASSEMBLY_RAM_COPIES)
    dram_cap = min(dram_onnx, reassembly)
    storage_cap = MAX_STORAGE_BYTES / float(DISK_MODEL_COPIES)
    expected_bytes = factory_scaled_pair_bytes(float(budget.expected_ms))
    timeout_bytes = factory_scaled_pair_bytes(float(budget.timeout_ms))
    frame_bytes = factory_scaled_pair_bytes(period)
    return (
        SizeCeiling("4 ms expected", expected_bytes, "latency"),
        SizeCeiling("20 ms FDIR timeout", timeout_bytes, "latency"),
        SizeCeiling("35 Hz frame", frame_bytes, "latency"),
        SizeCeiling("daily uplink", float(MAX_DAILY_UPLINK_BYTES), "uplink"),
        SizeCeiling("DRAM loadable", dram_cap, "dram"),
        SizeCeiling("storage (3 copies)", storage_cap, "storage"),
    )


def binding_usable_ceiling() -> SizeCeiling:
    """Return the tightest ceiling that still runs inside the FDIR timeout.

    Returns:
        The 20 ms factory-scale latency cap. Uplink and DRAM are larger.
    """
    timeout = size_ceilings()[1]
    return timeout


def catalog_precision(kind: str) -> str:
    """Return the quality-knee precision for a catalog kind.

    Args:
        kind: ``cls`` or ``seg``.

    Returns:
        ``fp16`` for classifiers, ``int8`` for segmentors.

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    if kind == "cls":
        return CLS_FLIGHT_PRECISION
    if kind == "seg":
        return SEG_FLIGHT_PRECISION
    raise ValueError(f"unknown kind {kind!r}")


def catalog_fit(arch: CatalogArch) -> CatalogFit:
    """Return full-frame wall time of ``arch`` beside the shipped partner.

    Args:
        arch: Stage-2 catalog row (128-px FLOPs).

    Returns:
        CatalogFit with pair detect and the three time-budget flags.
    """
    precision = catalog_precision(arch.kind)
    flops_g = arch.flops_128_g * catalog_area_scale()
    nbytes = bytes_moved_for(float(arch.params), precision)
    wall = model_latency(flops_g, nbytes, precision).wall_ms
    if arch.kind == "cls":
        pair = wall + seg_latency("int8").wall_ms
    else:
        pair = cls_latency("fp16").wall_ms + wall
    budget = derive_budget()
    period = frame_period_ms()
    return CatalogFit(
        arch=arch,
        precision=precision,
        flops_ff_g=flops_g,
        wall_ms=wall,
        pair_detect_ms=pair,
        onnx_bytes=params_bytes(float(arch.params), precision),
        fits_expected=pair <= float(budget.expected_ms),
        fits_timeout=pair <= float(budget.timeout_ms),
        fits_frame=pair <= period,
    )


def catalog_fits() -> tuple[CatalogFit, ...]:
    """Return CatalogFit for every locked catalog architecture."""
    return tuple(catalog_fit(arch) for arch in CATALOG)


@dataclass(frozen=True, slots=True)
class TensorWorkingSet:
    """Unified-DRAM footprint of both factory nets, resident together.

    Jetson has no discrete VRAM. These bytes come out of the same 8 GB LPDDR5
    pool as the OS and the Python flight process.

    Attributes:
        cls_weight_bytes: Classifier FP16 parameters.
        seg_weight_bytes: Segmentor INT8 parameters.
        input_bytes: NCHW float32 band plane.
        activation_bytes: Locked stride-4 live maps.
        cls_output_bytes: (1, 1) float32 logit.
        seg_output_bytes: (1, 1, H, W) float32 mask.
        onnx_bytes: Shipped factory files on disk.
        tensors_bytes: Weights plus I/O plus maps, both nets resident.
        workspace_bytes: Named ORT/TRT workspace reservation.
        gpu_held_bytes: Tensors plus workspace plus TRT engine estimate.
        frac_dram_tensors: ``tensors_bytes / 8 GB``.
        frac_dram_gpu_held: ``gpu_held_bytes / 8 GB``.
    """

    cls_weight_bytes: float
    seg_weight_bytes: float
    input_bytes: float
    activation_bytes: float
    cls_output_bytes: float
    seg_output_bytes: float
    onnx_bytes: float
    tensors_bytes: float
    workspace_bytes: float
    gpu_held_bytes: float
    frac_dram_tensors: float
    frac_dram_gpu_held: float


@dataclass(frozen=True, slots=True)
class BoardCompare:
    """One Orin SKU relative to the locked Nano Super.

    Attributes:
        spec: Locked board constants.
        cpu_x: (cores * GHz) over Nano Super.
        gpu_x: (CUDA cores * MHz) over Nano Super.
        dram_x: DRAM capacity over Nano Super.
        bw_x: DRAM bandwidth over Nano Super.
        tdp_x: Max module TDP over Nano Super.
        tdp_vs_payload: Module TDP over the 55 W payload-bus FDIR cap.
    """

    spec: BoardSpec
    cpu_x: float
    gpu_x: float
    dram_x: float
    bw_x: float
    tdp_x: float
    tdp_vs_payload: float


def tensor_working_set() -> TensorWorkingSet:
    """Return both factory nets resident in unified LPDDR5.

    Returns:
        TensorWorkingSet. Weights are FP16 classifier plus INT8 segmentor.
        TRT engines are ``TRT_ENGINE_OVERHEAD * onnx``. Workspace is the
        named 1 GB reservation, not a measured CUDA context.
    """
    cls_w = weight_bytes("cls", "fp16")
    seg_w = weight_bytes("seg", "int8")
    inp = input_bytes()
    act = activation_bytes()
    cls_out = cls_output_bytes()
    seg_out = seg_output_bytes()
    onnx = pair_onnx_bytes()
    tensors = cls_w + seg_w + inp + act + cls_out + seg_out
    workspace = ORT_WORKSPACE_GB * 1e9
    engines = TRT_ENGINE_OVERHEAD * onnx
    gpu_held = tensors + workspace + engines
    dram = DRAM_GB * 1e9
    return TensorWorkingSet(
        cls_weight_bytes=cls_w,
        seg_weight_bytes=seg_w,
        input_bytes=inp,
        activation_bytes=act,
        cls_output_bytes=cls_out,
        seg_output_bytes=seg_out,
        onnx_bytes=onnx,
        tensors_bytes=tensors,
        workspace_bytes=workspace,
        gpu_held_bytes=gpu_held,
        frac_dram_tensors=tensors / dram,
        frac_dram_gpu_held=gpu_held / dram,
    )


def _nano_cpu() -> float:
    """Return Nano Super core-GHz product."""
    return float(NANO_SUPER.cpu_cores) * NANO_SUPER.cpu_ghz


def _nano_gpu() -> float:
    """Return Nano Super CUDA-core-MHz product."""
    return float(NANO_SUPER.cuda_cores) * NANO_SUPER.gpu_mhz


def board_compare(spec: BoardSpec) -> BoardCompare:
    """Return ``spec`` scaled to the locked Nano Super.

    Args:
        spec: Nano Super, AGX 32GB, or AGX 64GB.

    Returns:
        BoardCompare with CPU/GPU/DRAM/bandwidth/TDP ratios.
    """
    cpu = float(spec.cpu_cores) * spec.cpu_ghz
    gpu = float(spec.cuda_cores) * spec.gpu_mhz
    return BoardCompare(
        spec=spec,
        cpu_x=cpu / _nano_cpu(),
        gpu_x=gpu / _nano_gpu(),
        dram_x=spec.dram_gb / NANO_SUPER.dram_gb,
        bw_x=spec.bw_gbps / NANO_SUPER.bw_gbps,
        tdp_x=spec.tdp_w / NANO_SUPER.tdp_w,
        tdp_vs_payload=spec.tdp_w / PAYLOAD_BUS_W,
    )


def board_compares() -> tuple[BoardCompare, ...]:
    """Return Nano Super, AGX 32GB, and AGX 64GB relative to Nano Super."""
    return tuple(board_compare(spec) for spec in COMPARE_BOARDS)
