"""Figures for the single-axis vs dual-axis gimbal study.

Geometry plots consume a GeometryResult. Industry plots consume cluster
lists, the TimeLostFn interpolator, and a RadiusProfile. matplotlib is
forced to Agg so the study can run without a display.

Contains:
  - plot_along_track, plot_disk_angle, plot_time_vs_radius, plot_lost_time.
  - plot_footprint, plot_offset_map, plot_required_az, plot_latitude.
  - plot_lat_hist, plot_folded, plot_expected_vs_lat, plot_r_hist, plot_map.
  - plot_r_vs_lat.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.lib.plot_style import C_BLUE, C_INK, C_ORANGE, C_RED, C_SKY, C_TEAL
from analysis.lib.tracking import TimeLostFn, angular_rate_deg_s
from analysis.studies.single_axis_vs_dual_axis_gimbal.assumptions import (
    LAT_BIN_DEG,
    MAX_HW_SLEW_DEG_S,
    PLUME_R_KM,
    TLE,
)
from analysis.studies.single_axis_vs_dual_axis_gimbal.inventory import Cluster
from analysis.studies.single_axis_vs_dual_axis_gimbal.profile import RadiusProfile

if TYPE_CHECKING:
    from analysis.studies.single_axis_vs_dual_axis_gimbal.geometry import GeometryResult


def plot_along_track(result: GeometryResult, path: Path) -> None:
    """Write elevation, rate, and origin |az| vs time for the design pass.

    Args:
        result: Design-pass tables and origin samples.
        path: Output PNG path.

    Returns:
        None.
    """
    times = result.times
    data = result.origin_samples
    optics = result.optics
    box = result.box
    lat_deg = TLE.inclination_deg
    t_s = data.t
    mask = (t_s >= times.t_start_s - 15) & (t_s <= times.t_stop_s + 15)
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.4), sharex=True)
    axes[0].plot(t_s[mask], data.el[mask], color=C_BLUE, lw=2)
    axes[0].axhline(box.el_nadir_deg, color=C_INK, ls=":", lw=1)
    axes[0].axhline(box.el_limb_deg, color=C_ORANGE, ls="--", lw=1, label="el stop 30 deg")
    axes[0].axvspan(
        times.t_start_s, times.t_stop_s, color=C_BLUE, alpha=0.12, label="elevation window"
    )
    axes[0].set_ylabel("elevation (deg)")
    axes[0].set_ylim(20, 100)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title(
        f"Along-track track time = {times.along_track_s:.1f} s "
        f"(lat {lat_deg:.1f} deg, h = {times.local_alt_km:.0f} km, one-sided 90->30)"
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
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(
        t_s[mask], np.abs(data.az[mask]), color=C_TEAL, lw=2, label="|az| of cluster origin"
    )
    axes[2].axhline(optics.half_az_deg, color=C_INK, ls="--", label="sensor half-FOV")
    axes[2].set_ylabel("|az| (deg)")
    axes[2].set_xlabel("time from closest approach (s)")
    axes[2].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_disk_angle(result: GeometryResult, path: Path) -> None:
    """Write worst-case covering-disk |az| vs time at the design pass.

    Args:
        result: Design-pass tables.
        path: Output PNG path.

    Returns:
        None.
    """
    from analysis.studies.single_axis_vs_dual_axis_gimbal.geometry import disk_max_az

    times = result.times
    t_s = np.arange(times.t_start_s, times.t_stop_s + 1.0, 1.0)
    lat_deg = TLE.inclination_deg
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    radii = (1.0, 2.0, 5.0, 8.0, 10.0, 15.0)
    colors = [C_SKY, C_BLUE, C_TEAL, C_ORANGE, C_RED, "#882255"]
    for radius, color in zip(radii, colors, strict=True):
        az = [disk_max_az(result.orbit, result.box, float(ti), lat_deg, radius) for ti in t_s]
        ax.plot(t_s, az, color=color, lw=2, label=f"R = {radius:.0f} km")
    ax.axhline(
        result.optics.half_az_deg,
        color=C_INK,
        ls="--",
        lw=1.4,
        label=f"sensor half-FOV {result.optics.half_az_deg:.2f} deg",
    )
    ax.set_xlabel("time from closest approach (s)")
    ax.set_ylabel("worst-case |az| of covering disk (deg)")
    ax.set_title(f"Covering-disk azimuth at lat {lat_deg:.1f} deg (Earth rotation on)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_time_vs_radius(result: GeometryResult, path: Path) -> None:
    """Write in-frame tracking time vs covering radius at the design pass.

    Args:
        result: Design-pass tables.
        path: Output PNG path.

    Returns:
        None.
    """
    table = result.table
    optics = result.optics
    h_km = result.times.local_alt_km
    lat_deg = TLE.inclination_deg
    radius = table.radius_km
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(
        radius,
        table.two_axis_s,
        color=C_BLUE,
        lw=2.4,
        marker="o",
        label="two-axis (box +/-10 deg az)",
    )
    ax.plot(
        radius, table.one_axis_s, color=C_ORANGE, lw=2.4, marker="s", label="one-axis (az parked)"
    )
    ax.axvline(
        h_km * math.tan(math.radians(optics.half_az_deg)),
        color=C_TEAL,
        ls="--",
        label="R = h tan(FOV_az / 2) at nadir",
    )
    ax.axvline(PLUME_R_KM, color=C_INK, ls=":", label=f"L = {PLUME_R_KM:.0f} km plume envelope")
    ax.set_xlabel("covering-disk radius R (km)")
    ax.set_ylabel("in-frame tracking time (s)")
    ax.set_title(f"Tracking time vs covering radius at lat {lat_deg:.1f} deg")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_lost_time(result: GeometryResult, path: Path) -> None:
    """Write time lost vs two-axis as a function of covering radius.

    Args:
        result: Design-pass tables.
        path: Output PNG path.

    Returns:
        None.
    """
    table = result.table
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(table.radius_km, table.lost_s, color=C_RED, lw=2.4, marker="o")
    ax.axvline(PLUME_R_KM, color=C_INK, ls=":", label=f"L = {PLUME_R_KM:.0f} km plume envelope")
    ax.set_xlabel("covering-disk radius R (km)")
    ax.set_ylabel("tracking time lost vs two-axis (s)")
    ax.set_title("Time given up by dropping the azimuth axis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_footprint(result: GeometryResult, path: Path) -> None:
    """Write sensor ground half-width vs time at the design pass.

    Args:
        result: Design-pass tables.
        path: Output PNG path.

    Returns:
        None.
    """
    from analysis.studies.single_axis_vs_dual_axis_gimbal.geometry import footprint_half_width_km

    times = result.times
    lat_deg = TLE.inclination_deg
    t_s = np.linspace(times.t_start_s, times.t_stop_s, 80)
    along: list[float] = []
    cross: list[float] = []
    for ti in t_s:
        a_km, c_km = footprint_half_width_km(
            result.orbit, result.optics, result.box, float(ti), lat_deg
        )
        along.append(a_km)
        cross.append(c_km)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(t_s, along, color=C_BLUE, lw=2, label="along-track half-footprint")
    ax.plot(t_s, cross, color=C_ORANGE, lw=2, label="cross-track half-footprint")
    ax.set_xlabel("time from closest approach (s)")
    ax.set_ylabel("ground half-width of sensor FOV (km)")
    ax.set_title(f"Sensor footprint at lat {lat_deg:.1f} deg")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_offset_map(result: GeometryResult, path: Path) -> None:
    """Write tracking time vs cross-track origin offset at the design pass.

    Args:
        result: Design-pass tables.
        path: Output PNG path.

    Returns:
        None.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ys = [row.origin_cross_km for row in result.offsets]
    ax.plot(
        ys,
        [row.two_axis_s for row in result.offsets],
        color=C_BLUE,
        lw=2.4,
        marker="o",
        label="two-axis",
    )
    ax.plot(
        ys,
        [row.one_axis_s for row in result.offsets],
        color=C_ORANGE,
        lw=2.4,
        marker="s",
        label="one-axis",
    )
    ax.set_xlabel("cluster origin cross-track offset (km)")
    ax.set_ylabel("in-frame tracking time (s)")
    ax.set_title("If the cluster is not on the slice centerline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_required_az(result: GeometryResult, path: Path) -> None:
    """Write peak disk-edge |az| vs covering radius.

    Args:
        result: Design-pass tables.
        path: Output PNG path.

    Returns:
        None.
    """
    table = result.table
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(table.radius_km, table.az_peak_deg, color=C_BLUE, lw=2.4, marker="o")
    ax.axhline(result.optics.half_az_deg, color=C_ORANGE, ls="--", label="1-axis: sensor half-FOV")
    ax.axhline(result.box.az_box_deg, color=C_INK, ls=":", label="2-axis operational az box")
    ax.set_xlabel("covering-disk radius R (km)")
    ax.set_ylabel("peak |az| during the elevation window (deg)")
    ax.set_title("Azimuth a 2-axis gimbal would use to keep the disk edge on boresight")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_latitude(result: GeometryResult, path: Path) -> None:
    """Write origin |az| and tracking time vs pass latitude.

    Args:
        result: Design-pass tables.
        path: Output PNG path.

    Returns:
        None.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.4), sharex=True)
    lat0 = [row.lat_deg for row in result.lat_r0]
    axes[0].plot(
        lat0,
        [row.az_max_deg for row in result.lat_r0],
        color=C_TEAL,
        lw=2.4,
        marker="o",
        label="R = 0 (origin)",
    )
    axes[0].axhline(result.optics.half_az_deg, color=C_INK, ls="--", label="sensor half-FOV")
    axes[0].set_ylabel("peak |az| of origin (deg)")
    axes[0].set_title("Earth rotation walks a nadir-centered origin in azimuth")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(
        lat0,
        [row.el_window_s for row in result.lat_r0],
        color=C_BLUE,
        lw=2,
        marker="o",
        label="elevation window (2-axis)",
    )
    axes[1].plot(
        lat0,
        [row.one_axis_s for row in result.lat_r0],
        color=C_ORANGE,
        lw=2,
        marker="s",
        label="1-axis, R = 0",
    )
    axes[1].set_xlabel("pass latitude (deg)")
    axes[1].set_ylabel("in-frame tracking time (s)")
    axes[1].legend(loc="lower right", fontsize=8)
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


def plot_folded(
    clusters: list[Cluster], fn: TimeLostFn, profile: RadiusProfile, path: Path
) -> None:
    """Write folded-|lat| stack mass against 1-axis and 2-axis tracking time.

    Args:
        clusters: World clusters; ISS-belt filter applied here.
        fn: Tracking-time interpolator.
        profile: R(|lat|) interpolator.
        path: Output PNG path.

    Returns:
        None.
    """
    iss_i = TLE.inclination_deg
    iss = [c for c in clusters if abs(c.lat) <= iss_i]
    abs_lat = np.array([abs(c.lat) for c in iss])
    w = np.array([float(c.n) for c in iss])
    edges = np.arange(0.0, iss_i + 2.0, 2.0)
    hist, _ = np.histogram(abs_lat, bins=edges, weights=w)
    centres = 0.5 * (edges[:-1] + edges[1:])
    t1 = np.zeros(centres.size)
    t2 = np.zeros(centres.size)
    for i, x in enumerate(centres):
        t1[i], t2[i], _ = fn.eval(float(x), profile.r_km(float(x)))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax2 = ax.twinx()
    ax2.bar(
        centres,
        100.0 * hist / max(1.0, float(np.sum(hist))),
        width=1.8,
        color=C_BLUE,
        alpha=0.3,
        label="stack %",
    )
    ax.plot(centres, t2, color=C_TEAL, lw=2, marker="o", ms=3, label="2-axis, R(|lat|)")
    ax.plot(centres, t1, color=C_RED, lw=2, marker="^", ms=3, label="1-axis, R(|lat|)")
    ax.set_xlabel("|latitude| (deg)")
    ax.set_ylabel("tracking time (s)")
    ax2.set_ylabel("ISS-belt stack fraction (%)")
    ax.set_title("Folded latitude: stack mass vs tracking time at R(|lat|)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_expected_vs_lat(
    edges: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    w: np.ndarray,
    path: Path,
) -> None:
    """Write tracking time vs latitude with stack-mass bars.

    Args:
        edges: Bin edges in degrees.
        t1: One-axis time at bin midpoints for R(|lat|).
        t2: Two-axis time at bin midpoints for R(|lat|).
        w: Stack counts per bin.
        path: Output PNG path.

    Returns:
        None.
    """
    centres = 0.5 * (edges[:-1] + edges[1:])
    iss_i = TLE.inclination_deg
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax2 = ax.twinx()
    ax2.bar(centres, w, width=LAT_BIN_DEG * 0.9, color=C_BLUE, alpha=0.25, label="ISS-belt stacks")
    ax.plot(centres, t2, color=C_TEAL, lw=2, marker="o", ms=3, label="2-axis, R(|lat|)")
    ax.plot(centres, t1, color=C_RED, lw=2, marker="^", ms=3, label="1-axis, R(|lat|)")
    ax.axvline(iss_i, color=C_RED, ls="--", lw=1)
    ax.axvline(-iss_i, color=C_RED, ls="--", lw=1)
    ax.set_xlabel("latitude (deg)")
    ax.set_ylabel("mean tracking time (s)")
    ax2.set_ylabel("stack-bearing sources in bin")
    ax.set_title("Tracking time vs latitude at R(|lat|), with stack mass")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_r_hist(clusters: list[Cluster], path: Path) -> None:
    """Write stack-weighted covering-radius histogram for the ISS belt.

    Args:
        clusters: World clusters; ISS-belt filter applied here.
        path: Output PNG path.

    Returns:
        None.
    """
    iss = [c for c in clusters if abs(c.lat) <= TLE.inclination_deg]
    radius = np.array([c.r_cover_km for c in iss])
    w = np.array([float(c.n) for c in iss])
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    bin_edges = np.linspace(0.0, 16.0, 33).tolist()
    ax.hist(radius, bins=bin_edges, weights=w, color=C_ORANGE, edgecolor="white")
    ax.axvline(
        PLUME_R_KM, color=C_INK, ls=":", label=f"L = {PLUME_R_KM:.0f} km (isolated-stack floor)"
    )
    ax.set_xlabel("covering radius R = D + L (km)")
    ax.set_ylabel("stack-bearing sources")
    ax.set_title("ISS-belt cluster size (stack-weighted)")
    ax.legend()
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
