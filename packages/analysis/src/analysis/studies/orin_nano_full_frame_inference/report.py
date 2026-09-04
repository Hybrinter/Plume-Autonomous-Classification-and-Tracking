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
    CAMERA_PREPROCESS_GB,
    CLS_ACC_FP32,
    CLS_ACC_INT8,
    CLS_ARCH,
    CLS_FLIGHT_ONNX_BYTES,
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
    MAX_DAILY_UPLINK_BYTES,
    MAX_FRAME_RATE_HZ,
    MODULE_TDP_W,
    NVPMODEL,
    ORT_WORKSPACE_GB,
    OS_RUNTIME_GB,
    PAYLOAD_BUS_W,
    PYTHON_JETPACK,
    PYTHON_REPO,
    SEG_ARCH,
    SEG_FLIGHT_ONNX_BYTES,
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
from analysis.studies.orin_nano_full_frame_inference.cost import (
    activation_bytes,
    cls_flops_ff_g,
    input_bytes,
    seg_flops_ff_g,
)
from analysis.studies.orin_nano_full_frame_inference.figures import write_figures
from analysis.studies.orin_nano_full_frame_inference.headroom import (
    binding_usable_ceiling,
    catalog_fits,
    factory_scale_for_detect,
    factory_scaled_pair_bytes,
    memory_floor_wall_ms,
    pair_onnx_bytes,
    size_ceilings,
    uplink_seconds,
    utilization,
)
from analysis.studies.orin_nano_full_frame_inference.latency import (
    PRECISIONS,
    cls_latency,
    derive_budget,
    duty_cycle,
    mixed_knee_detect_ms,
    seg_latency,
)


def _mib(nbytes: float) -> str:
    """Format bytes as MiB with two decimals.

    Args:
        nbytes: Size in bytes.

    Returns:
        Fixed two-decimal MiB string.
    """
    return f"{nbytes / (1024.0 * 1024.0):.2f}"


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
    occ_h = utilization()
    usable_h = binding_usable_ceiling()
    a(f"| GPU busy at 35 Hz detect | {occ_h.gpu_busy_frac_35hz * 100:.1f}% |")
    a(f"| **Usable pair at 20 ms** | **{_mib(usable_h.max_pair_bytes)} MiB** |")
    a(f"| Daily uplink cap | {_mib(float(MAX_DAILY_UPLINK_BYTES))} MiB |")
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
    a(
        f"| Shipped ONNX | {_mib(float(CLS_FLIGHT_ONNX_BYTES))} MiB FP16 | "
        f"{_mib(float(SEG_FLIGHT_ONNX_BYTES))} MiB INT8 |"
    )
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
    a("## Headroom")
    a("")
    occ = utilization()
    floor_ms = memory_floor_wall_ms()
    a("The 4 ms expected budget is ceil of the mixed-knee detect wall, so it")
    a("is tight by construction. Silicon and camera time are not.")
    a("")
    a("| Against | Used | Remaining |")
    a("| --- | --- | --- |")
    a(
        f"| Expected {occ.expected_ms} ms | {occ.frac_expected * 100:.1f}% "
        f"({occ.detect_wall_ms:.2f} ms) | "
        f"{(1.0 - occ.frac_expected) * 100:.1f}% |"
    )
    a(
        f"| FDIR timeout {occ.timeout_ms} ms | {occ.frac_timeout * 100:.1f}% | "
        f"{(1.0 - occ.frac_timeout) * 100:.1f}% |"
    )
    a(
        f"| {MAX_FRAME_RATE_HZ:g} Hz frame ({occ.frame_period_ms:.2f} ms) | "
        f"{occ.frac_frame * 100:.1f}% | {(1.0 - occ.frac_frame) * 100:.1f}% |"
    )
    a(
        f"| GPU kernel at 35 Hz detect | {occ.gpu_busy_frac_35hz * 100:.1f}% | "
        f"{(1.0 - occ.gpu_busy_frac_35hz) * 100:.1f}% |"
    )
    a("")
    a(f"Weightless memory-floor wall is {floor_ms:.2f} ms per net.")
    a(
        f"That is {_mib(input_bytes())} MiB float32 input plus "
        f"{_mib(activation_bytes())} MiB stride-4 maps at 102 GB/s, then wrap."
    )
    a(f"The shipped kernels ({occ.kernel_ms:.2f} ms together) sit on that floor.")
    a(f"Weights are 0.8 MiB against {_mib(occ.io_activation_bytes)} MiB of I/O and maps.")
    a(
        f"DRAM reservation leaves {occ.dram_free_gb:.2f} GB of {DRAM_GB:g} GB "
        f"after {OS_RUNTIME_GB:g} GB OS/runtime, {ORT_WORKSPACE_GB:g} GB ORT/TRT "
        f"workspace, and {CAMERA_PREPROCESS_GB:g} GB camera buffers."
    )
    a("That split is a policy, not a Nano measurement.")
    a("")
    a("## Max pair size")
    a("")
    a("A classifier+segmentor pair upload is one bundle. Factory-family size")
    a("scales linearly with detect wall while weights stay small versus maps.")
    a("Catalog rows keep the shipped partner and swap one net.")
    a("")
    a(f"Shipped pair on disk: {_mib(pair_onnx_bytes())} MiB.")
    a("")
    a("| Ceiling | Max pair | Binds |")
    a("| --- | --- | --- |")
    for ceiling in size_ceilings():
        a(f"| {ceiling.name} | {_mib(ceiling.max_pair_bytes)} MiB | {ceiling.binds} |")
    usable = binding_usable_ceiling()
    scale_timeout = factory_scale_for_detect(float(occ.timeout_ms))
    uplink_s = uplink_seconds(float(MAX_DAILY_UPLINK_BYTES))
    a("")
    a(
        f"Usable real-time ceiling is **{_mib(usable.max_pair_bytes)} MiB** "
        f"(factory family grown {scale_timeout:.2f}x to the {occ.timeout_ms} ms "
        "timeout)."
    )
    a("Raising expected/timeout is required past 4 ms.")
    a(f"Daily uplink is {_mib(float(MAX_DAILY_UPLINK_BYTES))} MiB ({uplink_s:.0f} s at 2 Mbps).")
    ratio = float(MAX_DAILY_UPLINK_BYTES) / factory_scaled_pair_bytes(float(occ.timeout_ms))
    a(f"That cap is {ratio:.0f}x the timeout-scaled factory pair.")
    a("A 100 MiB dense 1024x1224 convnet does not meet 20 ms.")
    a("Reassembly holds the blob in RAM with no other size cap.")
    a("CCSDS packets are 64 KiB, so large files are chunked.")
    a("")
    a("Catalog drop-in (128-px stage-2 FLOPs times 76.5). Classifiers FP16,")
    a("segmentors INT8, other net stays the shipped factory graph.")
    a("")
    a("| Arch | Kind | Params | FF GFLOP | Pair detect | 4 ms | 20 ms | 35 Hz |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in catalog_fits():
        yes = {True: "yes", False: "no"}
        a(
            f"| `{row.arch.name}` | {row.arch.kind} | "
            f"{row.arch.params / 1e6:.2f} M | {row.flops_ff_g:.2f} | "
            f"{row.pair_detect_ms:.2f} ms | {yes[row.fits_expected]} | "
            f"{yes[row.fits_timeout]} | {yes[row.fits_frame]} |"
        )
    a("")
    a("Largest catalog classifier inside 20 ms with the shipped segmentor is")
    a("`efficientnet_b0_pt` (~8 MiB FP16 weights, ~15 ms cls wall). ResNet-18")
    a("and larger miss the timeout. Largest catalog segmentor inside 20 ms")
    a("with the shipped classifier is `unet_w32_sep` (~0.4 MiB INT8). Baseline")
    a("U-Net and ResNet-50 are not real-time at 1024x1224 on this module.")
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
    a("![Time headroom](outputs/headroom_time.png)")
    a("")
    a("![Max pair size](outputs/max_pair_size.png)")
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
    a(
        "- Factory pair is 0.77 MiB. Real-time uploads bind on the 20 ms "
        "timeout (~5 MiB factory-family, ~8 MiB EfficientNet-B0), not the "
        "100 MiB daily uplink."
    )
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
