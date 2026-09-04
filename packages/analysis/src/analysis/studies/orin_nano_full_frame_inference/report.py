"""Markdown reports for the Orin Nano Super full-frame inference study.

Generated artifacts, not STE descriptive docs.

Contains:
  - write_results / write_readme / write_all.
"""

from __future__ import annotations

from pathlib import Path

from analysis.studies.orin_nano_full_frame_inference.assumptions import (
    BOARD_NAME,
    BW_GBPS,
    CAMERA_BUS,
    CAMERA_NAME,
    CLS_ACC_FP32,
    CLS_ACC_INT8,
    CLS_ARCH,
    CLS_FLOPS_TILE_G,
    CLS_PARAMS,
    CUDA_CORES,
    DRAM_GB,
    ETA_COMPUTE,
    ETA_WRAP,
    FLIGHT_PRECISION,
    FP16_TFLOPS,
    FP32_TFLOPS,
    FRAME_H_PX,
    FRAME_W_PX,
    HAS_DLA,
    INT8_DENSE_TOPS,
    INT8_SPARSE_TOPS,
    JETPACK,
    JETSON_CLOCKS,
    MODULE_TDP_W,
    NVPMODEL,
    PAYLOAD_BUS_W,
    PYTHON_JETPACK,
    PYTHON_REPO,
    SEG_ARCH,
    SEG_FLOPS_TILE_G,
    SEG_IOU_FP32,
    SEG_IOU_INT8,
    SEG_PARAMS,
    STUDY_DIR,
    TENSOR_CORES,
    TILE_H_PX,
    TILE_W_PX,
    TIMEOUT_MULT,
    area_scale,
)
from analysis.studies.orin_nano_full_frame_inference.cost import cls_flops_ff_g, seg_flops_ff_g
from analysis.studies.orin_nano_full_frame_inference.figures import write_figures
from analysis.studies.orin_nano_full_frame_inference.latency import (
    PRECISIONS,
    cls_latency,
    derive_budget,
    duty_cycle,
    mixed_knee_detect_ms,
    seg_latency,
)


def write_results(path: Path | None = None) -> Path:
    """Write RESULTS.md with the derived expected and timeout pair.

    Args:
        path: Destination markdown path. Default is STUDY_DIR / RESULTS.md.

    Returns:
        Written path.
    """
    dest = path if path is not None else STUDY_DIR / "RESULTS.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    budget = derive_budget()
    scale = area_scale()
    lines: list[str] = []
    a = lines.append
    a("# Orin Nano Super full-frame inference")
    a("")
    a("TEMPORARY ANALYSIS. Not flight software.")
    a("")
    a("This study derives the flight expected-detect latency and the FDIR")
    a("inference timeout from Orin Nano Super peaks, model FLOPs, and a named")
    a("efficiency policy. It does not use placeholder millisecond targets.")
    a("")
    a("## Headline")
    a("")
    a("| Field | Value |")
    a("| --- | --- |")
    a(f"| Board | {BOARD_NAME} |")
    a(f"| Shipped precision | {FLIGHT_PRECISION} |")
    a(f"| Detect wall (analytic) | {budget.t_detect_ms:.2f} ms |")
    a(f"| **Expected latency** | **{budget.expected_ms} ms** |")
    a(f"| **FDIR timeout** | **{budget.timeout_ms} ms** |")
    a(f"| Timeout policy | `ceil({TIMEOUT_MULT:g} x expected)` |")
    a("")
    a("Write `expected_ms` into `inference.latency_budget_ms` and `timeout_ms`")
    a("into `fault.inference_timeout_ms`.")
    a("")
    a("## Board lock")
    a("")
    a(f"- Ampere {CUDA_CORES} CUDA / {TENSOR_CORES} tensor cores.")
    a(f"- {FP16_TFLOPS:g} FP16 TFLOPS, {FP32_TFLOPS:g} FP32 TFLOPS (CUDA cores).")
    a(f"- {INT8_DENSE_TOPS:g} dense / {INT8_SPARSE_TOPS:g} sparse INT8 TOPS.")
    a(f"- {DRAM_GB:g} GB LPDDR5 at {BW_GBPS:g} GB/s.")
    a(f"- Module Super TDP {MODULE_TDP_W:g} W. Payload-bus FDIR stays {PAYLOAD_BUS_W:g} W.")
    a(f"- DLA present: {HAS_DLA}.")
    a(f"- Camera: {CAMERA_NAME} on {CAMERA_BUS} (no CSI path in this repo).")
    a(f"- Power mode for a later Nano CSV: `{NVPMODEL}` + `jetson_clocks`")
    a(f"  ({JETSON_CLOCKS}), JetPack {JETPACK}.")
    a(f"- Repo Python {PYTHON_REPO}; JetPack system Python {PYTHON_JETPACK}.")
    a("  This study does not change `requires-python`.")
    a("")
    a("## Models")
    a("")
    a("| | Classifier | Segmentor |")
    a("| --- | --- | --- |")
    a(f"| Arch | `{CLS_ARCH}` | `{SEG_ARCH}` |")
    a(f"| Params | {CLS_PARAMS / 1000:.0f} k | {SEG_PARAMS / 1000:.1f} k |")
    a(f"| FLOPs @ {TILE_H_PX}x{TILE_W_PX} | {CLS_FLOPS_TILE_G:.3f} G | {SEG_FLOPS_TILE_G:.2f} G |")
    a(
        f"| FLOPs @ {FRAME_H_PX}x{FRAME_W_PX} | {cls_flops_ff_g():.3f} G | "
        f"{seg_flops_ff_g():.2f} G |"
    )
    a(f"| Area scale | {scale:.3f} | {scale:.3f} |")
    a(f"| 256-tile FP32 quality | acc {CLS_ACC_FP32:.3f} | IoU {SEG_IOU_FP32:.3f} |")
    a(f"| 256-tile INT8 quality | acc {CLS_ACC_INT8:.3f} | IoU {SEG_IOU_INT8:.3f} |")
    a("")
    a("Re-export at 1024x1224 is a contract fix. It does not create a full-frame")
    a("IoU number. Quality remains the 256-tile stage-3 measurement.")
    a("")
    a("## Duty cycle")
    a("")
    a("Classifier runs every frame. Segmentor runs only on a positive presence")
    a("decision. `t_search = t_cls`. `t_detect = t_cls + t_seg`.")
    a("")
    a("| Precision | t_search (ms) | t_detect (ms) | cls kernel | seg kernel |")
    a("| --- | --- | --- | --- | --- |")
    for precision in PRECISIONS:
        duty = duty_cycle(precision)
        cls_k = cls_latency(precision)
        seg_k = seg_latency(precision)
        a(
            f"| {precision} | {duty.t_search_ms:.2f} | {duty.t_detect_ms:.2f} | "
            f"{cls_k.kernel_ms:.2f} | {seg_k.kernel_ms:.2f} |"
        )
    a("")
    a("## Latency model")
    a("")
    a("- Compute bound: `GFLOP / peak_TFLOPS`.")
    a(f"- Memory bound: bytes / ({BW_GBPS:g} GB/s).")
    a(f"- Kernel: `max(compute / {ETA_COMPUTE:g}, memory)`.")
    a(f"- Wall: `kernel / {ETA_WRAP:g}` (ORT + Python wrap).")
    a("- `expected_ms = ceil(t_detect)` for the shipped mixed knee.")
    a(f"- `timeout_ms = ceil({TIMEOUT_MULT:g} * expected_ms)`.")
    a("")
    a("## Quantization knee")
    a("")
    a("- Segmentor INT8 keeps 256-tile IoU. INT8 is the segmentor quality knee.")
    a("- Classifier INT8 drops accuracy 0.980 to 0.932. FP16 is the classifier")
    a("  working knee (same accuracy as FP32, Super tensor cores).")
    a(f"- Mixed knee detect wall (cls FP16 + seg INT8): {mixed_knee_detect_ms():.2f} ms.")
    a("- Factory ONNX is that pair: classifier FP16, segmentor INT8 QDQ.")
    a("  Graph I/O stay float32. `use_int8` stays false because the configured")
    a("  paths already point at the quantized graphs, not FP32 siblings.")
    a("")
    a("## Figures")
    a("")
    a("![Duty-cycle latency](outputs/duty_cycle_latency.png)")
    a("")
    a("![Quantization Pareto](outputs/quantization_pareto.png)")
    a("")
    a("![Artifact size](outputs/artifact_size.png)")
    a("")
    a("![FLOPs vs spatial size](outputs/flops_vs_spatial.png)")
    a("")
    a("![Mean time vs plume-positive rate](outputs/mean_time_vs_positive_rate.png)")
    a("")
    a("CPU ORT traces are not plotted as Orin. A laptop-GPU CSV, when present,")
    a("is labelled separately from Super estimates.")
    a("")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def write_readme(path: Path | None = None) -> Path:
    """Write the study folder README.md.

    Args:
        path: Destination path. Default is STUDY_DIR / README.md.

    Returns:
        Written path.
    """
    dest = path if path is not None else STUDY_DIR / "README.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    budget = derive_budget()
    lines: list[str] = []
    a = lines.append
    a("# Orin Nano Super full-frame inference")
    a("")
    a("TEMPORARY ANALYSIS. Not flight software. Design study for full-frame")
    a(f"1024x1224 inference on {BOARD_NAME}.")
    a("")
    a("```text")
    a("uv run python -m analysis.studies.orin_nano_full_frame_inference all")
    a("uv run python -m analysis.studies.orin_nano_full_frame_inference bench")
    a("```")
    a("")
    a("Geometry and budgets: [`RESULTS.md`](RESULTS.md). Figures: [`outputs/`](outputs/).")
    a("")
    a("## Ops lock")
    a("")
    a(f"- Power mode: `{NVPMODEL}` at module TDP {MODULE_TDP_W:g} W, then `jetson_clocks`.")
    a("  HIL procedure records the same mode when Nano traces exist.")
    a(f"- Payload-bus FDIR `power_limit_w` stays {PAYLOAD_BUS_W:g} W. That is not module TDP.")
    a(f"- Camera is {CAMERA_NAME} over {CAMERA_BUS}. There is no MIPI/CSI path.")
    a("- No DLA on this module. Inference is GPU / CUDA cores + tensor cores.")
    a(f"- Repo Python is {PYTHON_REPO}. JetPack {JETPACK} system Python is {PYTHON_JETPACK}.")
    a("  The flight image does not change `requires-python` for this gap.")
    a(f"- Derived expected {budget.expected_ms} ms, timeout {budget.timeout_ms} ms (mixed knee).")
    a("")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def write_all() -> tuple[Path, Path, list[Path]]:
    """Write figures, RESULTS.md, and README.md.

    Returns:
        (results_path, readme_path, figure_paths).
    """
    figures = write_figures()
    results = write_results()
    readme = write_readme()
    return results, readme, figures
