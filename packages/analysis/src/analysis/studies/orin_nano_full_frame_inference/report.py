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
    CPU_CORES,
    CPU_GHZ,
    CPU_NAME,
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
    board_compares,
    catalog_fits,
    factory_scale_for_detect,
    factory_scaled_pair_bytes,
    memory_floor_wall_ms,
    pair_onnx_bytes,
    size_ceilings,
    tensor_working_set,
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
    mem = tensor_working_set()
    a(f"| **Both-model tensors** | **{_mib(mem.tensors_bytes)} MiB** |")
    a(
        f"| Weights (cls FP16 + seg INT8) | "
        f"{_mib(mem.cls_weight_bytes + mem.seg_weight_bytes)} MiB |"
    )
    a(f"| Tensors / 8 GB unified DRAM | {mem.frac_dram_tensors * 100:.2f}% |")
    a(f"| **Usable pair at 20 ms** | **{_mib(usable_h.max_pair_bytes)} MiB** |")
    a(f"| Daily uplink cap | {_mib(float(MAX_DAILY_UPLINK_BYTES))} MiB |")
    a("")
    a("Write `expected_ms` into `inference.latency_budget_ms` and `timeout_ms`")
    a("into `fault.inference_timeout_ms`.")
    a("")
    a("## Board lock")
    a("")
    a(f"- Ampere {CUDA_CORES} CUDA / {TENSOR_CORES} tensor cores.")
    a(f"- CPU: {CPU_CORES}-core {CPU_NAME} at {CPU_GHZ:g} GHz. Unified LPDDR5, no VRAM.")
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
    a("## Unified memory (not VRAM)")
    a("")
    mem = tensor_working_set()
    a("Jetson Orin has no discrete VRAM. CPU, GPU, camera, and models share")
    a(f"{DRAM_GB:g} GB of LPDDR5.")
    a("")
    a("| Resident | Bytes |")
    a("| --- | --- |")
    a(f"| Classifier FP16 weights | {_mib(mem.cls_weight_bytes)} MiB |")
    a(f"| Segmentor INT8 weights | {_mib(mem.seg_weight_bytes)} MiB |")
    a(f"| Shipped ONNX pair | {_mib(mem.onnx_bytes)} MiB |")
    a(f"| Input (1, 4, 1024, 1224) float32 | {_mib(mem.input_bytes)} MiB |")
    a(f"| Live maps (stride 4, 64 ch) | {_mib(mem.activation_bytes)} MiB |")
    a(f"| Segmentor mask out | {_mib(mem.seg_output_bytes)} MiB |")
    a(f"| **Both nets, tensors** | **{_mib(mem.tensors_bytes)} MiB** |")
    a(
        f"| Plus 1 GB workspace + TRT engines | {_mib(mem.gpu_held_bytes)} MiB "
        f"({mem.frac_dram_gpu_held * 100:.1f}% of 8 GB) |"
    )
    a("")
    a(
        f"Weights are {_mib(mem.cls_weight_bytes + mem.seg_weight_bytes)} MiB. "
        "Almost all of the inference footprint is the float32 band plane and "
        "feature maps, not parameters."
    )
    a("If latency is not the gate, the 100 MiB daily uplink binds before DRAM.")
    a("A wide full-res U-Net can grow skip maps toward ~1 GB and still fit.")
    a("ResNet-50 FP16 (~45 MiB) plus the shipped segmentor is a rounding error.")
    a("")
    a("## Nano vs AGX")
    a("")
    a(f"Every Orin SKU uses {CPU_NAME}. AGX is more cores and a higher clock,")
    a("not a faster CPU microarchitecture.")
    a(f"Nano Super is {CPU_CORES} cores at {CPU_GHZ:g} GHz.")
    a("")
    a("| | CPU | GPU | DRAM | BW | TDP | TDP / 55 W bus | DLA |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for cmp in board_compares():
        spec = cmp.spec
        a(
            f"| {spec.name} | {spec.cpu_cores}c x {spec.cpu_ghz:g} GHz "
            f"({cmp.cpu_x:.2f}x) | {spec.cuda_cores} CUDA ({cmp.gpu_x:.2f}x) | "
            f"{spec.dram_gb:g} GB ({cmp.dram_x:.2f}x) | {spec.bw_gbps:g} GB/s "
            f"({cmp.bw_x:.2f}x) | {spec.tdp_w:g} W ({cmp.tdp_x:.2f}x) | "
            f"{cmp.tdp_vs_payload * 100:.0f}% | {spec.dla_count} |"
        )
    a("")
    a("AGX 64GB is about 2.6x CPU throughput and 2.5x GPU CUDA-clock product.")
    a("It is 8x DRAM capacity and 2x bandwidth. It is not 10x compute.")
    a(f"AGX max TDP {60:g} W exceeds payload-bus FDIR {PAYLOAD_BUS_W:g} W.")
    a("Nano Super 25 W is 45% of that bus. The flight stack is Python apps")
    a("on six A78AE cores; extra cores do not change the model working set.")
    a("Orin NX 16GB Super is the SODIMM step if DRAM ever binds: 8 cores,")
    a("16 GB, 102 GB/s, 2x DLA, 25-40 W. AGX is the wrong lever for this pair.")
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
    for fit in catalog_fits():
        yes = {True: "yes", False: "no"}
        a(
            f"| `{fit.arch.name}` | {fit.arch.kind} | "
            f"{fit.arch.params / 1e6:.2f} M | {fit.flops_ff_g:.2f} | "
            f"{fit.pair_detect_ms:.2f} ms | {yes[fit.fits_expected]} | "
            f"{yes[fit.fits_timeout]} | {yes[fit.fits_frame]} |"
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
    a("![Unified DRAM stack](outputs/dram_stack.png)")
    a("")
    a("![Nano vs AGX](outputs/nano_vs_agx.png)")
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
    a("- Factory pair tensors are ~82 MiB of unified LPDDR5 (no discrete VRAM).")
    a(
        "- AGX uses the same Cortex-A78AE CPU (8 or 12 cores vs 6). It is not "
        "10x compute; max TDP 60 W exceeds the 55 W payload-bus FDIR cap."
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
