"""Figures for the Orin Nano Super full-frame inference study.

None of the axes or annotations cite 100 ms or 500 ms.

Contains:
  - write_figures: nine PNGs under OUT_DIR.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from analysis.lib.plot_style import C_BLUE, C_ORANGE, C_RED, C_SKY, C_TEAL, apply
from analysis.studies.orin_nano_full_frame_inference.assumptions import (
    CAMERA_PREPROCESS_GB,
    CLS_ACC_FP16,
    CLS_ACC_FP32,
    CLS_ACC_INT8,
    CLS_FLOPS_TILE_G,
    CLS_FP32_BYTES,
    CLS_INT8_BYTES,
    DRAM_GB,
    ORT_WORKSPACE_GB,
    OS_RUNTIME_GB,
    OUT_DIR,
    SEG_FLOPS_TILE_G,
    SEG_FP32_BYTES,
    SEG_INT8_BYTES,
    SEG_IOU_FP16,
    SEG_IOU_FP32,
    SEG_IOU_INT8,
    area_scale,
)
from analysis.studies.orin_nano_full_frame_inference.cost import cls_flops_ff_g, seg_flops_ff_g
from analysis.studies.orin_nano_full_frame_inference.headroom import (
    board_compares,
    pair_onnx_bytes,
    size_ceilings,
    tensor_working_set,
    utilization,
)
from analysis.studies.orin_nano_full_frame_inference.latency import (
    PRECISIONS,
    derive_budget,
    duty_cycle,
    load_bench_csv,
    mean_time_ms,
    mixed_knee_detect_ms,
)


def _save(fig: Figure, name: str) -> Path:
    """Save ``fig`` under OUT_DIR and close it."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_duty_cycle() -> Path:
    """Grouped bars: Search vs Detect for FP32 / FP16 / INT8."""
    apply()
    labels = ["FP32", "FP16", "INT8"]
    search = [duty_cycle(p).t_search_ms for p in PRECISIONS]
    detect = [duty_cycle(p).t_detect_ms for p in PRECISIONS]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - width / 2, search, width, label="Search (classifier)", color=C_BLUE)
    ax.bar(x + width / 2, detect, width, label="Detect (cls+seg)", color=C_ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Analytic wall time (ms)")
    ax.set_title("Orin Nano Super duty-cycle latency")
    ax.legend()
    return _save(fig, "duty_cycle_latency.png")


def fig_quantization_pareto() -> Path:
    """256-tile quality vs analytic full-frame detect milliseconds."""
    apply()
    cls_pts = [
        ("FP32", CLS_ACC_FP32, duty_cycle("fp32").t_detect_ms, C_BLUE),
        ("FP16", CLS_ACC_FP16, duty_cycle("fp16").t_detect_ms, C_TEAL),
        ("INT8", CLS_ACC_INT8, duty_cycle("int8").t_detect_ms, C_ORANGE),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    ax_c, ax_s = axes
    for name, acc, ms, color in cls_pts:
        ax_c.scatter([ms], [acc], s=80, color=color, zorder=3, label=name)
        ax_c.annotate(name, (ms, acc), textcoords="offset points", xytext=(6, 4))
    ax_c.scatter(
        [duty_cycle("fp16").t_detect_ms],
        [CLS_ACC_FP16],
        s=240,
        facecolors="none",
        edgecolors=C_TEAL,
        linewidths=2,
        zorder=2,
        label="knee",
    )
    ax_c.set_xlabel("Detect wall (ms, both nets at that precision)")
    ax_c.set_ylabel("Classifier accuracy (256-tile)")
    ax_c.set_title("Classifier quality vs latency")
    ax_c.set_ylim(0.90, 1.0)
    ax_c.legend()

    seg_pts = [
        ("FP32", SEG_IOU_FP32, duty_cycle("fp32").t_detect_ms, C_BLUE),
        ("FP16", SEG_IOU_FP16, duty_cycle("fp16").t_detect_ms, C_TEAL),
        ("INT8", SEG_IOU_INT8, duty_cycle("int8").t_detect_ms, C_ORANGE),
    ]
    for name, iou, ms, color in seg_pts:
        ax_s.scatter([ms], [iou], s=80, color=color, zorder=3, label=name)
        ax_s.annotate(name, (ms, iou), textcoords="offset points", xytext=(6, 4))
    ax_s.scatter(
        [duty_cycle("int8").t_detect_ms],
        [SEG_IOU_INT8],
        s=240,
        facecolors="none",
        edgecolors=C_ORANGE,
        linewidths=2,
        zorder=2,
        label="knee",
    )
    ax_s.set_xlabel("Detect wall (ms, both nets at that precision)")
    ax_s.set_ylabel("Segmentor IoU (256-tile)")
    ax_s.set_title("Segmentor quality vs latency")
    ax_s.set_ylim(0.52, 0.58)
    ax_s.legend()
    fig.suptitle("Quantization Pareto (quality from 256-tile stage 3)")
    fig.tight_layout()
    rows = load_bench_csv()
    gpu = [row for row in rows if row.get("host") == "laptop_gpu"]
    if gpu:
        fig.text(
            0.5,
            -0.02,
            "Laptop GPU ORT overlay is not an Orin Super measurement.",
            ha="center",
            fontsize=8,
        )
    return _save(fig, "quantization_pareto.png")


def fig_artifact_size() -> Path:
    """FP32 vs INT8 artifact bytes."""
    apply()
    labels = ["Classifier", "Segmentor"]
    fp32 = [CLS_FP32_BYTES / 1024, SEG_FP32_BYTES / 1024]
    int8 = [CLS_INT8_BYTES / 1024, SEG_INT8_BYTES / 1024]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(x - width / 2, fp32, width, label="FP32", color=C_BLUE)
    ax.bar(x + width / 2, int8, width, label="INT8", color=C_ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Artifact size (KiB)")
    ax.set_title("ONNX artifact size")
    ax.legend()
    return _save(fig, "artifact_size.png")


def fig_flops_vs_spatial() -> Path:
    """FLOPs at 256 vs 1024x1224."""
    apply()
    labels = ["Classifier", "Segmentor"]
    tile = [CLS_FLOPS_TILE_G, SEG_FLOPS_TILE_G]
    full = [cls_flops_ff_g(), seg_flops_ff_g()]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(x - width / 2, tile, width, label="256 x 256", color=C_BLUE)
    ax.bar(x + width / 2, full, width, label="1024 x 1224", color=C_ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("GFLOPs")
    ax.set_title(f"FLOPs vs spatial size (area scale {area_scale():.3f})")
    ax.legend()
    return _save(fig, "flops_vs_spatial.png")


def fig_mean_vs_positive_rate() -> Path:
    """Mean wall time versus plume-positive rate."""
    apply()
    rates = np.linspace(0.0, 1.0, 21)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for precision, color, label in (
        ("fp32", C_BLUE, "FP32"),
        ("fp16", C_TEAL, "FP16"),
        ("int8", C_ORANGE, "INT8"),
    ):
        ax.plot(rates, [mean_time_ms(precision, float(p)) for p in rates], color=color, label=label)
    mixed = mixed_knee_detect_ms()
    ax.axhline(mixed, color=C_RED, linestyle="--", linewidth=1.0, label="Knee cls FP16 + seg INT8")
    ax.set_xlabel("Plume-positive rate p")
    ax.set_ylabel("Mean wall time (ms)")
    ax.set_title(r"$t_{\mathrm{cls}} + p\, t_{\mathrm{seg}}$")
    ax.legend()
    budget = derive_budget()
    ax.text(
        0.02,
        0.98,
        f"Derived expected {budget.expected_ms} ms, timeout {budget.timeout_ms} ms (mixed knee)",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    return _save(fig, "mean_time_vs_positive_rate.png")


def fig_headroom_time() -> Path:
    """Horizontal bars: shipped detect vs expected, timeout, and 35 Hz frame."""
    apply()
    occ = utilization()
    labels = [
        "Detect wall",
        "Expected (4 ms)",
        "FDIR timeout",
        "35 Hz frame",
    ]
    values = [
        occ.detect_wall_ms,
        float(occ.expected_ms),
        float(occ.timeout_ms),
        occ.frame_period_ms,
    ]
    colors = [C_TEAL, C_BLUE, C_ORANGE, C_SKY]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Milliseconds")
    ax.set_title("Orin Nano Super time headroom")
    for idx, val in enumerate(values):
        ax.text(val + 0.4, idx, f"{val:.2f} ms", va="center", fontsize=9)
    ax.set_xlim(0, occ.frame_period_ms * 1.25)
    return _save(fig, "headroom_time.png")


def fig_max_pair_size() -> Path:
    """Log pair-size ceilings: factory scale, uplink, DRAM, storage."""
    apply()
    shipped_mib = pair_onnx_bytes() / (1024.0 * 1024.0)
    rows = [("shipped pair", shipped_mib, C_TEAL)]
    colors = (C_BLUE, C_BLUE, C_BLUE, C_ORANGE, C_SKY, C_RED)
    for ceiling, color in zip(size_ceilings(), colors, strict=True):
        rows.append((ceiling.name, ceiling.max_pair_bytes / (1024.0 * 1024.0), color))
    labels = [row[0] for row in rows]
    mib = [row[1] for row in rows]
    bar_colors = [row[2] for row in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    y = np.arange(len(labels))
    ax.barh(y, mib, color=bar_colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Pair artifact (MiB)")
    ax.set_title("Max classifier+segmentor pair size")
    ax.axvline(shipped_mib, color=C_TEAL, linestyle=":", linewidth=1.0)
    return _save(fig, "max_pair_size.png")


def fig_dram_stack() -> Path:
    """Stacked 8 GB bar: OS, workspace, camera, both-model tensors, free."""
    apply()
    ws = tensor_working_set()
    tensors_gb = ws.tensors_bytes / 1e9
    reserved = OS_RUNTIME_GB + ORT_WORKSPACE_GB + CAMERA_PREPROCESS_GB + tensors_gb
    chunks = (
        ("OS / runtime", OS_RUNTIME_GB, C_ORANGE),
        ("ORT/TRT workspace", ORT_WORKSPACE_GB, C_SKY),
        ("Camera / preprocess", CAMERA_PREPROCESS_GB, C_BLUE),
        ("Both-model tensors", tensors_gb, C_TEAL),
        ("Free", DRAM_GB - reserved, "#BBBBBB"),
    )
    fig, ax = plt.subplots(figsize=(7.4, 2.4))
    left = 0.0
    for name, width, color in chunks:
        ax.barh([0], [width], left=left, color=color, label=name)
        left += width
    ax.set_yticks([])
    ax.set_xlim(0, DRAM_GB)
    ax.set_xlabel("Unified LPDDR5 (GB)")
    ax.set_title("Orin Nano Super 8 GB: no discrete VRAM")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3, fontsize=8)
    return _save(fig, "dram_stack.png")


def fig_nano_vs_agx() -> Path:
    """Relative CPU, GPU, DRAM, bandwidth, and TDP vs Nano Super."""
    apply()
    rows = board_compares()
    labels = ["CPU", "GPU", "DRAM", "Bandwidth", "TDP"]
    series = [
        (
            row.spec.name.replace("Jetson ", ""),
            [row.cpu_x, row.gpu_x, row.dram_x, row.bw_x, row.tdp_x],
        )
        for row in rows
    ]
    colors = (C_TEAL, C_BLUE, C_ORANGE)
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for idx, (name, values) in enumerate(series):
        ax.bar(x + (idx - 1) * width, values, width, label=name, color=colors[idx])
    ax.axhline(1.0, color="#888888", linestyle=":", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Relative to Orin Nano Super")
    ax.set_title("Same Cortex-A78AE CPU; AGX is more cores, not a new CPU")
    ax.legend()
    return _save(fig, "nano_vs_agx.png")


def write_figures() -> list[Path]:
    """Write all study figures.

    Returns:
        Paths of the written PNGs.
    """
    return [
        fig_duty_cycle(),
        fig_quantization_pareto(),
        fig_artifact_size(),
        fig_flops_vs_spatial(),
        fig_mean_vs_positive_rate(),
        fig_headroom_time(),
        fig_max_pair_size(),
        fig_dram_stack(),
        fig_nano_vs_agx(),
    ]
