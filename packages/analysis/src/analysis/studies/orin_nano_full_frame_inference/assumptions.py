"""Locked assumptions for the Orin Nano Super full-frame inference study.

Board: Jetson Orin Nano Super 8 GB (P3766). Peaks are MAXN SUPER datasheet
values, not Maxwell Nano and not AGX Orin. The study never uses 100 ms or
500 ms as targets. Expected detect latency and the FDIR timeout are derived
from FLOPs, memory traffic, small-net efficiency, and wrap overhead.

Contains:
  - Super compute / memory / power constants.
  - Factory model sizes, 256-tile quality, and area scale.
  - Duty-cycle and budget-policy constants.
  - analysis_root / STUDY_DIR / DATA_DIR / OUT_DIR.
"""

from __future__ import annotations

from pathlib import Path

# --- Board (Orin Nano Super 8 GB, MAXN SUPER 25 W) ---
BOARD_NAME = "Jetson Orin Nano Super 8 GB"
CUDA_CORES = 1024
TENSOR_CORES = 32
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


def area_scale() -> float:
    """Return full-frame pixels divided by 256-tile pixels.

    Returns:
        Spatial area ratio (exact 19.125 for 1024x1224 over 256x256).
    """
    return (FRAME_H_PX * FRAME_W_PX) / (TILE_H_PX * TILE_W_PX)


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
