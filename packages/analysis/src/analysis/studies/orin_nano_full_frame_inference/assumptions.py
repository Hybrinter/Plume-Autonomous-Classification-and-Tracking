"""Locked assumptions for the Orin Nano Super full-frame inference study.

Board: Jetson Orin Nano Super 8 GB (P3766). Peaks are MAXN SUPER datasheet
values, not Maxwell Nano and not AGX Orin. The study never uses 100 ms or
500 ms as targets. Expected detect latency and the FDIR timeout are derived
from FLOPs, memory traffic, small-net efficiency, and wrap overhead.

Contains:
  - Super compute / memory / power constants.
  - Factory model sizes, 256-tile quality, and area scale.
  - Duty-cycle and budget-policy constants.
  - Headroom reservations (DRAM, uplink, camera rate) and catalog arches.
  - Nano Super vs AGX Orin board specs (NVIDIA public module table).
  - analysis_root / STUDY_DIR / DATA_DIR / OUT_DIR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Board (Orin Nano Super 8 GB, MAXN SUPER 25 W) ---
BOARD_NAME = "Jetson Orin Nano Super 8 GB"
CUDA_CORES = 1024
TENSOR_CORES = 32
GPU_MHZ = 1020.0
FP16_TFLOPS = 17.0
INT8_SPARSE_TOPS = 67.0
INT8_DENSE_TOPS = 33.0
FP32_TFLOPS = 2.08
BW_GBPS = 102.0
DRAM_GB = 8.0
HAS_DLA = False
MODULE_TDP_W = 25.0
PAYLOAD_BUS_W = 55.0
NVPMODEL = "MAXN SUPER"
JETSON_CLOCKS = True
CAMERA_BUS = "USB3"
CAMERA_NAME = "BFS-U3-50S5"
PYTHON_REPO = "3.14"
PYTHON_JETPACK = "3.10"
JETPACK = "6.2+"
CPU_NAME = "Arm Cortex-A78AE v8.2"
CPU_CORES = 6
CPU_GHZ = 1.7
CPU_L2_MB = 1.5
CPU_L3_MB = 4.0

# --- Spatial contract ---
TILE_H_PX = 256
TILE_W_PX = 256
FRAME_H_PX = 1024
FRAME_W_PX = 1224
IN_CHANNELS = 4

# --- Factory pair at 256 (stage-3 measured) ---
CLS_ARCH = "shufflenetv2_x0_5_pt"
SEG_ARCH = "dilatenet_w32"
CLS_PARAMS = 343_000
SEG_PARAMS = 23_500
CLS_FLOPS_TILE_G = 0.110
SEG_FLOPS_TILE_G = 0.21
CLS_FP32_BYTES = 1_350_000
SEG_FP32_BYTES = 96_000
CLS_INT8_BYTES = 540_000
SEG_INT8_BYTES = 47_000
# Shipped factory pair on disk (classifier FP16, segmentor INT8 QDQ).
CLS_FLIGHT_ONNX_BYTES = 763_679
SEG_FLIGHT_ONNX_BYTES = 44_141

CLS_ACC_FP32 = 0.980
CLS_ACC_FP16 = 0.980
CLS_ACC_INT8 = 0.932
SEG_IOU_FP32 = 0.553
SEG_IOU_FP16 = 0.553
SEG_IOU_INT8 = 0.553

# Live feature maps: three 64-channel tensors at output stride 4, float32.
LIVE_MAPS = 3
LIVE_CHANNELS = 64
OUTPUT_STRIDE = 4

# Small-net fraction of peak FLOPS (ShuffleNet / DilateNet on Ampere).
ETA_COMPUTE = 0.15
# ORT + Python wall time versus kernel time (efficiency < 1).
ETA_WRAP = 0.50
# FDIR missed-frame margin: timeout covers this many expected detect frames.
TIMEOUT_MULT = 5.0

# Flight ships the quality knee: classifier FP16, segmentor INT8.
FLIGHT_PRECISION = "mixed"
CLS_FLIGHT_PRECISION = "fp16"
SEG_FLIGHT_PRECISION = "int8"

PREFERRED_ORT_PROVIDERS: tuple[str, ...] = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)

CLS_ONNX = "data/models/active_classifier.onnx"
SEG_ONNX = "data/models/active_segmentor.onnx"

# Camera max rate from SensorConfig.max_frame_rate_hz (BFS-U3-50S5).
MAX_FRAME_RATE_HZ = 35.0

# Flight comms / storage caps (match CommsConfig / StorageConfig defaults).
MAX_DAILY_UPLINK_BYTES = 104_857_600  # 100 MiB
MAX_UPLINK_BPS = 2_000_000
MAX_STORAGE_BYTES = 107_374_182_400  # 100 GB placeholder
DISK_MODEL_COPIES = 3  # active + rollback + staged
REASSEMBLY_RAM_COPIES = 2  # chunk map plus concatenated blob

# DRAM reservation policy for a later Nano measurement. Not a live trace.
# 8 GB minus JetPack/Python/CUDA, ORT/TRT workspace, and camera buffers.
OS_RUNTIME_GB = 2.5
ORT_WORKSPACE_GB = 1.0
CAMERA_PREPROCESS_GB = 0.2
# TensorRT engine bytes relative to the ONNX file; two sessions during activate.
TRT_ENGINE_OVERHEAD = 2.0
GPU_SESSIONS_DURING_ACTIVATE = 2

# NVIDIA public Jetson Orin module table (not a Nano measurement).
# CPU is the same Cortex-A78AE on every Orin SKU. AGX is more cores and clock,
# not a different CPU. Memory is unified LPDDR5; there is no discrete VRAM.


@dataclass(frozen=True, slots=True)
class BoardSpec:
    """One Jetson Orin SKU for the Nano vs AGX comparison.

    Attributes:
        name: Module name.
        cpu_cores: Cortex-A78AE core count.
        cpu_ghz: Max CPU clock.
        cpu_l2_mb: Total L2.
        cpu_l3_mb: Total L3.
        cuda_cores: Ampere CUDA cores.
        gpu_mhz: Max GPU clock.
        dram_gb: Unified LPDDR5 capacity.
        bw_gbps: DRAM bandwidth.
        tdp_w: Max module TDP.
        dla_count: NVDLA v2 engines (0 on Nano).
    """

    name: str
    cpu_cores: int
    cpu_ghz: float
    cpu_l2_mb: float
    cpu_l3_mb: float
    cuda_cores: int
    gpu_mhz: float
    dram_gb: float
    bw_gbps: float
    tdp_w: float
    dla_count: int


NANO_SUPER = BoardSpec(
    name=BOARD_NAME,
    cpu_cores=CPU_CORES,
    cpu_ghz=CPU_GHZ,
    cpu_l2_mb=CPU_L2_MB,
    cpu_l3_mb=CPU_L3_MB,
    cuda_cores=CUDA_CORES,
    gpu_mhz=GPU_MHZ,
    dram_gb=DRAM_GB,
    bw_gbps=BW_GBPS,
    tdp_w=MODULE_TDP_W,
    dla_count=0,
)
AGX_ORIN_32GB = BoardSpec(
    name="Jetson AGX Orin 32GB",
    cpu_cores=8,
    cpu_ghz=2.2,
    cpu_l2_mb=2.0,
    cpu_l3_mb=4.0,
    cuda_cores=1792,
    gpu_mhz=1300.0,
    dram_gb=32.0,
    bw_gbps=204.8,
    tdp_w=60.0,
    dla_count=2,
)
AGX_ORIN_64GB = BoardSpec(
    name="Jetson AGX Orin 64GB",
    cpu_cores=12,
    cpu_ghz=2.2,
    cpu_l2_mb=3.0,
    cpu_l3_mb=6.0,
    cuda_cores=2048,
    gpu_mhz=1300.0,
    dram_gb=64.0,
    bw_gbps=204.8,
    tdp_w=60.0,
    dla_count=2,
)
COMPARE_BOARDS: tuple[BoardSpec, ...] = (NANO_SUPER, AGX_ORIN_32GB, AGX_ORIN_64GB)

# Stage-2 catalog at 128 px, four bands (experiments/results README).
CATALOG_TILE_PX = 128


@dataclass(frozen=True, slots=True)
class CatalogArch:
    """One stage-2 architecture with 128-px FLOPs, scaled later to 1024x1224.

    Attributes:
        name: Architecture id from the training catalog.
        kind: ``cls`` or ``seg``.
        params: Parameter count.
        flops_128_g: Forward G count at 128 x 128 x 4.
    """

    name: str
    kind: str
    params: int
    flops_128_g: float


# Classifier candidates scored as FP16; segmentors as INT8 (quality knee).
CATALOG: tuple[CatalogArch, ...] = (
    CatalogArch("shufflenetv2_x0_5_pt", "cls", 343_033, 0.03),
    CatalogArch("mobilenetv3_small_pt", "cls", 1_519_025, 0.04),
    CatalogArch("mobilenetv3_large_pt", "cls", 4_203_457, 0.15),
    CatalogArch("efficientnet_b0_pt", "cls", 4_009_117, 0.25),
    CatalogArch("resnet18_pt", "cls", 11_180_161, 1.21),
    CatalogArch("resnet50_pt", "cls", 23_513_217, 2.69),
    CatalogArch("dilatenet_w32", "seg", 23_521, 0.05),
    CatalogArch("dilatenet_w64", "seg", 83_905, 0.18),
    CatalogArch("unet_w16_sep", "seg", 105_973, 0.15),
    CatalogArch("unet_w32_sep", "seg", 397_765, 0.51),
    CatalogArch("unet_w8", "seg", 210_377, 0.25),
    CatalogArch("unet_w16", "seg", 838_929, 0.98),
    CatalogArch("unet_w32", "seg", 3_350_561, 3.89),
    CatalogArch("unet_baseline", "seg", 13_391_937, 15.48),
)


def area_scale() -> float:
    """Return full-frame pixels divided by 256-tile pixels.

    Returns:
        Spatial area ratio (exact 19.125 for 1024x1224 over 256x256).
    """
    return (FRAME_H_PX * FRAME_W_PX) / (TILE_H_PX * TILE_W_PX)


def catalog_area_scale() -> float:
    """Return full-frame pixels divided by the 128-px catalog tile.

    Returns:
        Spatial area ratio (exact 76.5 for 1024x1224 over 128x128).
    """
    tile = CATALOG_TILE_PX * CATALOG_TILE_PX
    return (FRAME_H_PX * FRAME_W_PX) / float(tile)


def analysis_root() -> Path:
    """Return the analysis workspace-member root (contains pyproject.toml).

    Returns:
        Path to packages/analysis/.

    Raises:
        RuntimeError: If this file is not under the analysis member.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "analysis").is_dir():
            return parent
    raise RuntimeError("could not find packages/analysis/ member root")


STUDY_DIR = analysis_root() / "orin_nano_full_frame_inference"
DATA_DIR = STUDY_DIR / "data"
OUT_DIR = STUDY_DIR / "outputs"
BENCH_CSV = DATA_DIR / "ort_bench.csv"
