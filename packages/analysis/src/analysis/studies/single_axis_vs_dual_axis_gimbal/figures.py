"""Figures for the single-axis vs dual-axis gimbal study.

Geometry plots consume a GeometryResult. Industry plots consume cluster
lists, the TimeLostFn interpolator, and a RadiusProfile. matplotlib is
forced to Agg so the study can run without a display.

Contains:
  - plot_along_track, plot_az_walk, plot_offset_map.
  - plot_lat_hist, plot_reacquire_vs_lat, plot_yield_vs_lat.
  - plot_hunt_timeline, plot_r_vs_lat, plot_map.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.lib.hunt import HuntResult
from analysis.lib.plot_style import C_BLUE, C_INK, C_ORANGE, C_RED, C_TEAL
from analysis.lib.tracking import angular_rate_deg_s, in_science_window
from analysis.studies.single_axis_vs_dual_axis_gimbal.assumptions import (
    AZ_WALK_LATS_DEG,
    LAT_BIN_DEG,
    MAX_HW_SLEW_DEG_S,
    OFFSET_PLANT_D_KM,
    OFFSET_STACK_N,
    PLUME_R_KM,
    POLAR_TASK_LAT_DEG,
    TLE,
)
from analysis.studies.single_axis_vs_dual_axis_gimbal.inventory import Cluster
from analysis.studies.single_axis_vs_dual_axis_gimbal.profile import RadiusProfile

if TYPE_CHECKING:
    from analysis.studies.single_axis_vs_dual_axis_gimbal.geometry import GeometryResult


def _iss_xlim() -> tuple[float, float]:
    """Return x-limits padded around the ISS inclination belt.

    Returns:
        (lo, hi) in degrees.
    """
    pad = 1.0
    return (-TLE.inclination_deg - pad, TLE.inclination_deg + pad)


def plot_along_track(result: GeometryResult, path: Path) -> None:
    """Write elevation and rate vs time for the design pass.

    Args:
        result: Design-pass tables and origin samples.
        path: Output PNG path.

    Returns:
        None.
    """
    times = result.times
    data = result.origin_samples
    box = result.box
    lat_deg = TLE.inclination_deg
    t_s = data.t
    mask = (t_s >= times.t_start_s - 15) & (t_s <= times.t_stop_s + 15)
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True)
    axes[0].plot(t_s[mask], data.el[mask], color=C_BLUE, lw=2)
    axes[0].axhline(box.el_nadir_deg, color=C_INK, ls=":", lw=1)
    axes[0].axhline(
        box.el_limb_deg,
        color=C_ORANGE,
        ls="--",
        lw=1,
        label=f"el stop {box.el_limb_deg:.0f} deg",
    )
    axes[0].axvspan(
        times.t_start_s, times.t_stop_s, color=C_BLUE, alpha=0.12, label="elevation window"
    )
    axes[0].set_ylabel("elevation (deg)")
    axes[0].set_ylim(20, 100)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title(
        f"Along-track track time = {times.along_track_s:.1f} s "
        f"(lat {lat_deg:.1f} deg, h = {times.local_alt_km:.0f} km, "
        f"one-sided 90->{box.el_limb_deg:.0f})"
    )

    rate = np.abs(angular_rate_deg_s(t_s, data.el))
    axes[1].plot(t_s[mask], rate[mask], color=C_ORANGE, lw=2)
    axes[1].axhline(
        result.omega_img_rewind_deg_s,
        color=C_TEAL,
        ls="--",
        label=f"imaging rewind {result.omega_img_rewind_deg_s:.2f} deg/s",
    )
    axes[1].axhline(
        MAX_HW_SLEW_DEG_S,
        color=C_RED,
        ls=":",
        label=f"hardware cap {MAX_HW_SLEW_DEG_S:.0f} deg/s",
    )
    axes[1].set_ylabel("|d(el)/dt| (deg/s)")
    axes[1].set_xlabel("time from closest approach (s)")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_az_walk(result: GeometryResult, path: Path) -> None:
    """Write origin |az| vs time at equator, mid-lat, and the design pass.

    Earth rotation walks a nadir origin in azimuth at low latitude. At
    max latitude the heading is due east and the walk vanishes.

    Args:
        result: Design-pass orbit, optics, and box.
        path: Output PNG path.

    Returns:
        None.
    """
    from analysis.studies.single_axis_vs_dual_axis_gimbal.geometry import origin_window

    colors = (C_ORANGE, C_TEAL, C_BLUE)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for lat_deg, color in zip(AZ_WALK_LATS_DEG, colors, strict=True):
        times, data = origin_window(result.orbit, result.optics, result.box, lat_deg, 0.0)
        mask = (data.t >= times.t_start_s - 5.0) & (data.t <= times.t_stop_s + 5.0)
        if not np.any(mask):
            mask = in_science_window(data, result.box)
        ax.plot(
            data.t[mask],
            np.abs(data.az[mask]),
            color=color,
            lw=2,
            label=f"lat {lat_deg:.1f} deg",
        )
    ax.axhline(
        result.optics.half_az_deg,
        color=C_INK,
        ls="--",
        lw=1.4,
        label=f"sensor half-FOV {result.optics.half_az_deg:.2f} deg",
    )
    ax.set_xlabel("time from closest approach (s)")
    ax.set_ylabel("|az| of nadir origin (deg)")
    ax.set_title("Earth rotation walks the origin in azimuth except at max latitude")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_offset_map(result: GeometryResult, path: Path) -> None:
    """Write plume-seconds and in-frame plume count vs cluster offset.

    A plume counts when half its L-disk is on the chip. One-axis parks at
    az = 0. Two-axis boresights each plume inside the keep-out box.

    Args:
        result: Design-pass offset rows.
        path: Output PNG path.

    Returns:
        None.
    """
    ys = [row.origin_cross_km for row in result.offsets]
    h_km = result.times.local_alt_km
    chip_km = h_km * math.tan(math.radians(result.optics.half_az_deg))
    box_km = h_km * math.tan(math.radians(result.box.az_box_deg))
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.4), sharex=True)
    axes[0].plot(
        ys,
        [row.two_axis_plume_s for row in result.offsets],
        color=C_TEAL,
        lw=2.4,
        marker="o",
        ms=3,
        label="2-axis",
    )
    axes[0].plot(
        ys,
        [row.one_axis_plume_s for row in result.offsets],
        color=C_RED,
        lw=2.4,
        marker="s",
        ms=3,
        label="1-axis",
    )
    for ax in axes:
        ax.axvline(chip_km, color=C_INK, ls="--", lw=1, label=f"chip half-swath {chip_km:.0f} km")
        ax.axvline(box_km, color=C_INK, ls=":", lw=1, label=f"box half-swath {box_km:.0f} km")
    axes[0].set_ylabel("plume-seconds")
    axes[0].set_title(
        f"Off-track information, n={OFFSET_STACK_N} stacks, "
        f"D={OFFSET_PLANT_D_KM:.1f} km, L={PLUME_R_KM:.0f} km"
    )
    axes[0].legend(loc="center right", fontsize=8)

    axes[1].step(
        ys,
        [row.two_axis_n_in for row in result.offsets],
        color=C_TEAL,
        lw=2.4,
        where="mid",
        label="2-axis plumes in frame",
    )
    axes[1].step(
        ys,
        [row.one_axis_n_in for row in result.offsets],
        color=C_RED,
        lw=2.4,
        where="mid",
        label="1-axis plumes in frame",
    )
    axes[1].set_xlabel("cluster centroid cross-track offset (km)")
    axes[1].set_ylabel("plumes with in-frame time")
    axes[1].set_ylim(-0.2, OFFSET_STACK_N + 0.4)
    axes[1].legend(loc="center right", fontsize=8)
    axes[1].set_xlim(0.0, 80.0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_lat_hist(
    edges: np.ndarray,
    h_area: np.ndarray,
    h_stacks: np.ndarray,
    h_em: np.ndarray,
    h_gppd: np.ndarray | None,
    path: Path,
) -> None:
    """Write stack-area, stack-count, and CO2 histograms vs latitude.

    Args:
        edges: Bin edges in degrees. # np.ndarray[float64, (B+1,)]
        h_area: Covering-disk area per bin.
        h_stacks: Stack counts per bin.
        h_em: 2025 CO2 tonnes per bin.
        h_gppd: Optional GPPD thermal counts, scaled onto the stack axis.
        path: Output PNG path.

    Returns:
        None.
    """
    centres = 0.5 * (edges[:-1] + edges[1:])
    iss_i = TLE.inclination_deg
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.5), sharex=True)
    specs = (
        (h_area, "covering-disk area (km2)", C_BLUE),
        (h_stacks, "stack-bearing sources (count)", C_ORANGE),
        (h_em, "2025 CO2 (tonnes)", C_TEAL),
    )
    for ax, (hist, ylab, col) in zip(axes, specs, strict=True):
        ax.bar(centres, hist, width=LAT_BIN_DEG * 0.9, color=col, align="center")
        ax.axvline(iss_i, color=C_RED, ls="--", lw=1.2, label=f"ISS max lat +/-{iss_i:.1f} deg")
        ax.axvline(-iss_i, color=C_RED, ls="--", lw=1.2)
        ax.axvline(45.0, color=C_INK, ls=":", lw=1)
        ax.axvline(-45.0, color=C_INK, ls=":")
        ax.set_ylabel(ylab)
        ax.legend(loc="upper right", fontsize=8)
    if h_gppd is not None and float(np.sum(h_gppd)) > 0:
        scale = float(np.sum(h_stacks)) / max(1.0, float(np.sum(h_gppd)))
        axes[1].plot(centres, h_gppd * scale, color=C_INK, lw=1.2, label="GPPD thermal (scaled)")
        axes[1].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("latitude (deg)")
    axes[0].set_title("Stack-bearing industrial areas vs latitude")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_reacquire_vs_lat(
    edges: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    reacq1: np.ndarray,
    reacq2: np.ndarray,
    w: np.ndarray,
    path: Path,
) -> None:
    """Write single-target dwell, reacquire, and cycle time vs latitude.

    Cycle time is dwell plus reacquire. The x-axis is the ISS belt. Stack
    counts sit on the dwell panel only.

    Args:
        edges: Bin edges in degrees.
        t1: One-axis single-target dwell at bin midpoints.
        t2: Two-axis single-target dwell at bin midpoints.
        reacq1: One-axis mean reacquire time.
        reacq2: Two-axis mean reacquire time.
        w: Stack counts per bin.
        path: Output PNG path.

    Returns:
        None.
    """
    centres = 0.5 * (edges[:-1] + edges[1:])
    iss_i = TLE.inclination_deg
    iss = np.abs(centres) <= iss_i + 0.05
    finite1 = iss & np.isfinite(reacq1)
    finite2 = iss & np.isfinite(reacq2)
    cyc1 = t1 + reacq1
    cyc2 = t2 + reacq2
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 10.2), sharex=True)
    xlim = _iss_xlim()

    ax0b = axes[0].twinx()
    ax0b.bar(
        centres[iss],
        w[iss],
        width=LAT_BIN_DEG * 0.9,
        color=C_BLUE,
        alpha=0.2,
        label="ISS-belt stacks",
    )
    axes[0].plot(centres[iss], t2[iss], color=C_TEAL, lw=2, marker="o", ms=3, label="2-axis dwell")
    axes[0].plot(centres[iss], t1[iss], color=C_RED, lw=2, marker="^", ms=3, label="1-axis dwell")
    axes[0].set_ylabel("dwell on one plant (s)")
    ax0b.set_ylabel("stack-bearing sources in bin")
    axes[0].set_title("How long one plant stays in the science window")
    h0, l0 = axes[0].get_legend_handles_labels()
    h0b, l0b = ax0b.get_legend_handles_labels()
    axes[0].legend(h0 + h0b, l0 + l0b, loc="center right", fontsize=8)

    axes[1].plot(
        centres[finite2],
        reacq2[finite2],
        color=C_TEAL,
        lw=2,
        marker="o",
        ms=3,
        label="2-axis",
    )
    axes[1].plot(
        centres[finite1],
        reacq1[finite1],
        color=C_RED,
        lw=2,
        marker="^",
        ms=3,
        label="1-axis",
    )
    axes[1].set_ylabel("mean wait to the next plant (s)")
    axes[1].set_yscale("log")
    axes[1].set_title("Lost time from signed-latitude stack density")
    axes[1].legend(loc="upper right", fontsize=8)

    cyc2_ok = iss & np.isfinite(cyc2)
    cyc1_ok = iss & np.isfinite(cyc1)
    axes[2].plot(
        centres[cyc2_ok], cyc2[cyc2_ok], color=C_TEAL, lw=2, marker="o", ms=3, label="2-axis"
    )
    axes[2].plot(
        centres[cyc1_ok], cyc1[cyc1_ok], color=C_RED, lw=2, marker="^", ms=3, label="1-axis"
    )
    axes[2].set_ylabel("start of one track to start of the next (s)")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("latitude (deg)")
    axes[2].set_title("Cycle = dwell + reacquire")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_xlim(*xlim)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_yield_vs_lat(
    edges: np.ndarray,
    yld1: np.ndarray,
    yld2: np.ndarray,
    w: np.ndarray,
    polar_1: float,
    belt_2: float,
    path: Path,
) -> None:
    """Write daily usable yield vs latitude and the polar vs belt bar.

    Yield is T_usable x daily coverage. The bar panel compares a 1-axis
    payload tasked only at |lat| >= 45 deg against 2-axis over the full
    ISS belt, both stack-weighted.

    Args:
        edges: Bin edges in degrees.
        yld1: One-axis T_usable x coverage at bin midpoints.
        yld2: Two-axis T_usable x coverage at bin midpoints.
        w: Stack counts per bin.
        polar_1: Stack-weighted 1-axis yield for |lat| >= 45 deg.
        belt_2: Stack-weighted 2-axis yield over the ISS belt.
        path: Output PNG path.

    Returns:
        None.
    """
    centres = 0.5 * (edges[:-1] + edges[1:])
    iss_i = TLE.inclination_deg
    iss = np.abs(centres) <= iss_i + 0.05
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.2), gridspec_kw={"height_ratios": (2.2, 1.0)})
    ax0b = axes[0].twinx()
    ax0b.bar(
        centres[iss],
        w[iss],
        width=LAT_BIN_DEG * 0.9,
        color=C_BLUE,
        alpha=0.2,
        label="ISS-belt stacks",
    )
    axes[0].plot(
        centres[iss], yld2[iss], color=C_TEAL, lw=2, marker="o", ms=3, label="2-axis yield"
    )
    axes[0].plot(centres[iss], yld1[iss], color=C_RED, lw=2, marker="^", ms=3, label="1-axis yield")
    axes[0].set_ylabel("daily usable yield (s x frac)")
    ax0b.set_ylabel("stack-bearing sources in bin")
    axes[0].set_title("T_usable x daily coverage vs latitude")
    axes[0].set_xlim(*_iss_xlim())
    h0, l0 = axes[0].get_legend_handles_labels()
    h0b, l0b = ax0b.get_legend_handles_labels()
    axes[0].legend(h0 + h0b, l0 + l0b, loc="upper right", fontsize=8)

    labels = (f"1-axis, |lat|>= {POLAR_TASK_LAT_DEG:.0f} deg", "2-axis, full ISS belt")
    vals = (polar_1, belt_2)
    axes[1].bar((0, 1), vals, color=(C_RED, C_TEAL), width=0.55)
    axes[1].set_xticks((0, 1), labels)
    axes[1].set_ylabel("stack-weighted yield")
    axes[1].set_title("Polar-tasked 1-axis vs full-belt 2-axis")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_hunt_timeline(
    rows: list[tuple[float, float, float, HuntResult]],
    path: Path,
) -> None:
    """Write a hunt Gantt at the industrial-belt latitudes.

    Each row is one gimbal at one signed latitude: TRACK dwell, elevation
    rewind, then the remainder of T_reacq (limb FOV wait or box raster).

    Args:
        rows: (lat_deg, t_dwell_1, t_dwell_2, HuntResult) per latitude.
        path: Output PNG path.

    Returns:
        None.
    """
    labels: list[str] = []
    series: list[tuple[float, float, float, str]] = []
    for lat_deg, t1, t2, hunted in rows:
        hem = "N" if lat_deg >= 0.0 else "S"
        labels.append(f"{abs(lat_deg):.0f} {hem}  2-axis")
        series.append((t2, hunted.t_rewind_2_s, hunted.t_reacq_2_s, C_TEAL))
        labels.append(f"{abs(lat_deg):.0f} {hem}  1-axis")
        series.append((t1, hunted.t_rewind_1_s, hunted.t_reacq_1_s, C_RED))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y_pos = np.arange(len(series))
    for y, (dwell, rewind, reacq, color) in zip(y_pos, series, strict=True):
        rewind_s = rewind if math.isfinite(rewind) else 0.0
        reacq_s = reacq if math.isfinite(reacq) else rewind_s
        limb = max(0.0, reacq_s - rewind_s)
        ax.barh(y, dwell, color=color, height=0.55)
        ax.barh(y, rewind_s, left=dwell, color=color, alpha=0.55, height=0.55)
        ax.barh(y, limb, left=dwell + rewind_s, color=color, alpha=0.25, height=0.55)
        if not math.isfinite(reacq):
            ax.annotate(
                "inf",
                (dwell + rewind_s + 2.0, y),
                va="center",
                fontsize=8,
                color=C_INK,
            )
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("seconds from start of track")
    ax.set_title("Hunt cycle: dwell, elevation rewind, then limb search")
    ax.plot([], [], color=C_INK, lw=8, label="dwell")
    ax.plot([], [], color=C_INK, lw=8, alpha=0.55, label="rewind")
    ax.plot([], [], color=C_INK, lw=8, alpha=0.25, label="limb wait / raster")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_r_vs_lat(profile: RadiusProfile, path: Path) -> None:
    """Write stack-weighted mean D and R vs |latitude|.

    Args:
        profile: Folded |lat| covering-radius profile.
        path: Output PNG path.

    Returns:
        None.
    """
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(
        profile.abs_lat,
        profile.mean_d_km,
        color=C_ORANGE,
        lw=2,
        marker="o",
        ms=3,
        label="mean D (plant span)",
    )
    ax.plot(
        profile.abs_lat,
        profile.mean_r_km,
        color=C_TEAL,
        lw=2,
        marker="s",
        ms=3,
        label="mean R = D + L",
    )
    ax.axhline(PLUME_R_KM, color=C_INK, ls=":", label=f"L = {PLUME_R_KM:.0f} km")
    ax.set_xlabel("|latitude| (deg)")
    ax.set_ylabel("stack-weighted radius (km)")
    ax.set_title("Covering radius vs latitude (no locked design R)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_map(clusters: list[Cluster], path: Path) -> None:
    """Write a lon/lat scatter of plant clusters.

    Marker area is proportional to covering-disk area. Colour is log10(n).

    Args:
        clusters: World clusters.
        path: Output PNG path.

    Returns:
        None.
    """
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    lats = np.array([c.lat for c in clusters])
    lons = np.array([c.lon for c in clusters])
    sizes = np.clip(np.array([c.area_km2 for c in clusters]) * 3.0, 2.0, 40.0)
    color = np.log10(np.maximum(np.array([c.n for c in clusters]), 1))
    ax.scatter(lons, lats, s=sizes, c=color, cmap="viridis", alpha=0.45, linewidths=0)
    ax.axhline(TLE.inclination_deg, color=C_RED, ls="--", lw=1)
    ax.axhline(-TLE.inclination_deg, color=C_RED, ls="--", lw=1)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title("Stack-bearing plant clusters (marker area ~ covering disk)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
