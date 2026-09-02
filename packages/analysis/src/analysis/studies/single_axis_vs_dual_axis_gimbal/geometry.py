"""Design-pass tracking-time tables wrapping analysis.lib.

Computes the one-sided elevation window, covering-disk azimuth, and
one-axis vs two-axis in-frame time vs radius, latitude, and origin
offset. ``run_geometry`` writes CSV, figures, and RESULTS.md.

Contains:
  - WindowTimes / RadiusTable / LatitudeRow / OffsetRow / GeometryResult.
  - origin_window / tracking_time_vs_radius / latitude_table / offset_times.
  - disk_max_az / footprint_half_width_km / self_check / run_geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from analysis.lib.constants import OMEGA_EARTH_RAD_S
from analysis.lib.hunt import HuntModel, HuntResult
from analysis.lib.look import GimbalBox, body_axes, earth_hit, look_at, rotate_z
from analysis.lib.optics import Optics, band_gsd_along_m, build_optics
from analysis.lib.orbit import (
    Orbit,
    argument_of_latitude,
    build_orbit,
    heading_from_north_deg,
    iss_eci,
    offset_ecef,
    origin_ecef,
)
from analysis.lib.plot_style import apply as apply_plot_style
from analysis.lib.tracking import (
    PassSamples,
    SampleSpan,
    disk_any_part_in_stop,
    in_science_window,
    mask_time_s,
    sample_pass,
    staring_in_frame,
    two_axis_boresightable,
)
from analysis.studies.single_axis_vs_dual_axis_gimbal.assumptions import (
    DESIGN_LAT_DEG,
    DISK_RADII_KM,
    GEOMETRY_DT_S,
    GIMBAL_BOX,
    GSD_MAX_BAND_M,
    MAX_HW_SLEW_DEG_S,
    OPTICS_SPEC,
    ORIGIN_OFFSETS_KM,
    OUT_DIR,
    PASS_LATS_DEG,
    SLANT_MAX_KM,
    STUDY_DIR,
    TLE,
    omega_img_rewind_deg_s,
    omega_rel_max_deg_s,
)


@dataclass
class WindowTimes:
    """Elevation-window summary for one origin sample.

    Attributes:
        along_track_s: Time from first to last in-window sample.
        stare_s: Time the origin stays in the staring FOV.
        peak_el_rate_deg_s: Peak |d(el)/dt| in the window.
        t_start_s: Window start from closest approach.
        t_stop_s: Window stop from closest approach.
        slant_start_km: Slant range at window start.
        slant_stop_km: Slant range at window stop.
        az_max_deg: Peak |az| of the origin in the window.
        local_alt_km: ISS altitude at the pass latitude.
        incidence_start_deg: Earth incidence at window start.
        gsd_band_along_start_m: Along-track band GSD at window start, metres.
    """

    along_track_s: float
    stare_s: float
    peak_el_rate_deg_s: float
    t_start_s: float
    t_stop_s: float
    slant_start_km: float
    slant_stop_km: float
    az_max_deg: float
    local_alt_km: float
    incidence_start_deg: float
    gsd_band_along_start_m: float


@dataclass
class RadiusTable:
    """Tracking time vs covering radius at one latitude.

    Attributes:
        radius_km: Covering radii. # np.ndarray[float64, (R,)]
        one_axis_s: One-axis in-frame time. # np.ndarray[float64, (R,)]
        two_axis_s: Two-axis in-frame time. # np.ndarray[float64, (R,)]
        lost_s: two_axis_s - one_axis_s. # np.ndarray[float64, (R,)]
        az_peak_deg: Peak |az| of the disk edge in the window.
    """

    radius_km: np.ndarray
    one_axis_s: np.ndarray
    two_axis_s: np.ndarray
    lost_s: np.ndarray
    az_peak_deg: np.ndarray


@dataclass(frozen=True)
class LatitudeRow:
    """One latitude in the Earth-rotation table."""

    lat_deg: float
    alt_km: float
    el_window_s: float
    az_max_deg: float
    one_axis_s: float
    two_axis_s: float


@dataclass(frozen=True)
class OffsetRow:
    """One cross-track origin offset at the design pass."""

    origin_cross_km: float
    one_axis_s: float
    two_axis_s: float


@dataclass
class GeometryResult:
    """All design-pass tables needed by the report and figures.

    Attributes:
        orbit: Circular ISS orbit (SMA).
        optics: Usable sensor FOV.
        box: Gimbal box.
        times: Design-pass window at SMA.
        origin_samples: Origin look-angle samples.
        times_perigee: Design-pass window at perigee radius.
        table: Tracking time vs covering radius at the design lat.
        offsets: Cross-track origin offsets at the design lat.
        lat_r0: Earth-rotation table for a point origin (R = 0).
        omega_rel_max_deg_s: 1 band-px / 1 ms scene-relative smear limit.
        omega_img_rewind_deg_s: Imaging rewind cap (rel max minus peak track).
        hunt: Rewind/scan diagnostics after a full-window loss at design lat.
    """

    orbit: Orbit
    optics: Optics
    box: GimbalBox
    times: WindowTimes
    origin_samples: PassSamples
    times_perigee: WindowTimes
    table: RadiusTable
    offsets: list[OffsetRow]
    lat_r0: list[LatitudeRow]
    omega_rel_max_deg_s: float
    omega_img_rewind_deg_s: float
    hunt: HuntResult


def _span() -> SampleSpan:
    """Return the design-pass sample grid."""
    return SampleSpan(dt_s=GEOMETRY_DT_S)


def origin_window(
    orbit: Orbit,
    optics: Optics,
    box: GimbalBox,
    lat_deg: float,
    origin_cross_km: float,
) -> tuple[WindowTimes, PassSamples]:
    """Return elevation-window times for a nadir-latitude origin, possibly offset.

    Args:
        orbit: Circular ISS orbit.
        optics: Usable sensor FOV.
        box: Gimbal box.
        lat_deg: Pass latitude in degrees.
        origin_cross_km: Cross-track offset of the origin in kilometres.

    Returns:
        Window summary and the origin pass samples.
    """
    data = sample_pass(orbit, box, lat_deg, 0.0, origin_cross_km, _span())
    el_mask = in_science_window(
        data, box, optics, slant_max_km=SLANT_MAX_KM, gsd_max_band_m=GSD_MAX_BAND_M
    )
    empty = WindowTimes(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if not np.any(el_mask):
        return empty, data
    t_s = data.t
    idx = np.where(el_mask)[0]
    t_start = float(t_s[idx[0]])
    t_stop = float(t_s[idx[-1]])
    stare = mask_time_s(staring_in_frame(data.az, data.el, optics, box), t_s)
    rate = np.abs(np.gradient(data.el, t_s))
    peak_rate = float(np.max(rate[el_mask])) if np.any(el_mask) else 0.0
    inc0 = float(data.incidence[idx[0]])
    slant0 = float(data.slant[idx[0]])
    return (
        WindowTimes(
            along_track_s=t_stop - t_start,
            stare_s=stare,
            peak_el_rate_deg_s=peak_rate,
            t_start_s=t_start,
            t_stop_s=t_stop,
            slant_start_km=slant0,
            slant_stop_km=float(data.slant[idx[-1]]),
            az_max_deg=float(np.max(np.abs(data.az[el_mask]))),
            local_alt_km=orbit.local_altitude_km(lat_deg),
            incidence_start_deg=inc0,
            gsd_band_along_start_m=band_gsd_along_m(optics.ifov_band_deg, slant0, inc0),
        ),
        data,
    )


def tracking_time_vs_radius(
    orbit: Orbit,
    optics: Optics,
    box: GimbalBox,
    lat_deg: float,
    radii: tuple[float, ...],
) -> RadiusTable:
    """Return in-frame time vs covering radius at one latitude.

    The disk is represented by the origin plus the +/- R cross-track edges.

    Args:
        orbit: Circular ISS orbit.
        optics: Usable sensor FOV.
        box: Gimbal box.
        lat_deg: Pass latitude in degrees.
        radii: Covering radii in kilometres.

    Returns:
        One-axis and two-axis times and peak disk-edge azimuth.
    """
    span = _span()
    r_arr = np.array(radii, dtype=float)
    t1 = np.zeros(r_arr.size)
    t2 = np.zeros(r_arr.size)
    az_peak = np.zeros(r_arr.size)
    origin = sample_pass(orbit, box, lat_deg, 0.0, 0.0, span)
    science = in_science_window(
        origin, box, optics, slant_max_km=SLANT_MAX_KM, gsd_max_band_m=GSD_MAX_BAND_M
    )
    for i, radius in enumerate(r_arr):
        edge_p = sample_pass(orbit, box, lat_deg, 0.0, float(radius), span)
        edge_m = sample_pass(orbit, box, lat_deg, 0.0, -float(radius), span)
        one = science & disk_any_part_in_stop(origin.az, edge_p.az, edge_m.az, optics.half_az_deg)
        two = science & disk_any_part_in_stop(origin.az, edge_p.az, edge_m.az, box.az_box_deg)
        t1[i] = mask_time_s(one, origin.t)
        t2[i] = mask_time_s(two, origin.t)
        az_peak[i] = (
            float(np.max(np.maximum(np.abs(edge_p.az), np.abs(edge_m.az))[science]))
            if np.any(science)
            else 0.0
        )
    return RadiusTable(
        radius_km=r_arr,
        one_axis_s=t1,
        two_axis_s=t2,
        lost_s=t2 - t1,
        az_peak_deg=az_peak,
    )


def latitude_table(
    orbit: Orbit,
    optics: Optics,
    box: GimbalBox,
    lats: tuple[float, ...],
    radius_km: float,
) -> list[LatitudeRow]:
    """Return tracking time vs pass latitude for one covering radius.

    Args:
        orbit: Circular ISS orbit.
        optics: Usable sensor FOV.
        box: Gimbal box.
        lats: Pass latitudes in degrees.
        radius_km: Covering-disk radius in kilometres.

    Returns:
        One row per latitude.
    """
    span = _span()
    rows: list[LatitudeRow] = []
    for lat in lats:
        times, origin = origin_window(orbit, optics, box, lat, 0.0)
        edge_p = sample_pass(orbit, box, lat, 0.0, radius_km, span)
        edge_m = sample_pass(orbit, box, lat, 0.0, -radius_km, span)
        science = in_science_window(
            origin, box, optics, slant_max_km=SLANT_MAX_KM, gsd_max_band_m=GSD_MAX_BAND_M
        )
        one = science & disk_any_part_in_stop(origin.az, edge_p.az, edge_m.az, optics.half_az_deg)
        two = science & disk_any_part_in_stop(origin.az, edge_p.az, edge_m.az, box.az_box_deg)
        rows.append(
            LatitudeRow(
                lat_deg=lat,
                alt_km=times.local_alt_km,
                el_window_s=times.along_track_s,
                az_max_deg=times.az_max_deg,
                one_axis_s=mask_time_s(one, origin.t),
                two_axis_s=mask_time_s(two, origin.t),
            )
        )
    return rows


def offset_times(
    orbit: Orbit,
    optics: Optics,
    box: GimbalBox,
    lat_deg: float,
) -> list[OffsetRow]:
    """Return tracking time vs cross-track origin offset at one latitude.

    Args:
        orbit: Circular ISS orbit.
        optics: Usable sensor FOV.
        box: Gimbal box.
        lat_deg: Pass latitude in degrees.

    Returns:
        One row per offset in ORIGIN_OFFSETS_KM.
    """
    rows: list[OffsetRow] = []
    for y_km in ORIGIN_OFFSETS_KM:
        _times, data = origin_window(orbit, optics, box, lat_deg, y_km)
        science = in_science_window(
            data, box, optics, slant_max_km=SLANT_MAX_KM, gsd_max_band_m=GSD_MAX_BAND_M
        )
        one = science & (np.abs(data.az) <= optics.half_az_deg)
        two = science & two_axis_boresightable(data.az, data.el, box)
        rows.append(
            OffsetRow(
                origin_cross_km=float(y_km),
                one_axis_s=mask_time_s(one, data.t),
                two_axis_s=mask_time_s(two, data.t),
            )
        )
    return rows


def disk_points(radius_km: float, n_pts: int = 72) -> np.ndarray:
    """Return (along, cross) samples on a circle of radius ``radius_km``.

    Args:
        radius_km: Circle radius in kilometres.
        n_pts: Number of samples around the circle.

    Returns:
        (N, 2) along/cross kilometres. # np.ndarray[float64, (N, 2)]
    """
    phis = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=False)
    return np.stack([radius_km * np.cos(phis), radius_km * np.sin(phis)], axis=1)


def disk_max_az(
    orbit: Orbit,
    box: GimbalBox,
    t_s: float,
    lat_deg: float,
    radius_km: float,
) -> float:
    """Return worst-case |az| of the covering disk at one epoch.

    Args:
        orbit: Circular ISS orbit.
        box: Gimbal box.
        t_s: Seconds from closest approach.
        lat_deg: Pass latitude in degrees.
        radius_km: Covering radius in kilometres.

    Returns:
        Max |az| in degrees over the disk boundary and origin.
    """
    u0 = argument_of_latitude(lat_deg, orbit.inclination_rad)
    heading = heading_from_north_deg(lat_deg, orbit.inclination_rad)
    origin = origin_ecef(orbit, lat_deg)
    r_iss, vel = iss_eci(t_s, orbit, u0)
    rot = rotate_z(OMEGA_EARTH_RAD_S * t_s)
    worst = 0.0
    for along, cross in disk_points(radius_km):
        tgt = rot @ offset_ecef(origin, lat_deg, float(along), float(cross), heading)
        worst = max(worst, abs(look_at(r_iss, vel, tgt, box.el_nadir_deg).az_deg))
    tgt0 = rot @ origin
    worst = max(worst, abs(look_at(r_iss, vel, tgt0, box.el_nadir_deg).az_deg))
    return worst


def footprint_half_width_km(
    orbit: Orbit,
    optics: Optics,
    box: GimbalBox,
    t_s: float,
    lat_deg: float,
) -> tuple[float, float]:
    """Return along-track and cross-track ground half-widths of the sensor FOV.

    Args:
        orbit: Circular ISS orbit.
        optics: Usable sensor FOV.
        box: Gimbal box.
        t_s: Seconds from closest approach.
        lat_deg: Pass latitude in degrees.

    Returns:
        (along_km, cross_km), or (nan, nan) if a ray misses the Earth sphere.
    """
    u0 = argument_of_latitude(lat_deg, orbit.inclination_rad)
    r_iss, vel = iss_eci(t_s, orbit, u0)
    origin = rotate_z(OMEGA_EARTH_RAD_S * t_s) @ origin_ecef(orbit, lat_deg)
    earth_r = float(np.linalg.norm(origin))
    x_axis, y_axis, _z_axis = body_axes(r_iss, vel)
    look0 = origin - r_iss
    u0v = look0 / np.linalg.norm(look0)
    half_el = math.radians(optics.half_el_deg)
    half_az = math.radians(optics.half_az_deg)

    def _rot(vec: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
        k_axis = axis / np.linalg.norm(axis)
        return cast(
            np.ndarray,
            vec * math.cos(ang)
            + np.cross(k_axis, vec) * math.sin(ang)
            + k_axis * np.dot(k_axis, vec) * (1.0 - math.cos(ang)),
        )

    p0 = earth_hit(r_iss, u0v, earth_r)
    p_el = earth_hit(r_iss, _rot(u0v, y_axis, half_el), earth_r)
    p_az = earth_hit(r_iss, _rot(u0v, x_axis, half_az), earth_r)
    if p0 is None or p_el is None or p_az is None:
        return float("nan"), float("nan")
    return float(np.linalg.norm(p_el - p0)), float(np.linalg.norm(p_az - p0))


def self_check(orbit: Orbit, optics: Optics, box: GimbalBox, times: WindowTimes) -> None:
    """Assert design-pass sanity: nadir look, window length, equatorial az walk.

    Args:
        orbit: Circular ISS orbit (SMA).
        optics: Usable sensor FOV.
        box: Gimbal box.
        times: Design-pass window.

    Returns:
        None.

    Raises:
        AssertionError: If a locked geometric check fails.
    """
    u0 = argument_of_latitude(DESIGN_LAT_DEG, orbit.inclination_rad)
    r_iss, vel = iss_eci(0.0, orbit, u0)
    tgt = origin_ecef(orbit, DESIGN_LAT_DEG)
    look0 = look_at(r_iss, vel, tgt, box.el_nadir_deg)
    assert abs(look0.el_deg - 90.0) < 0.05, look0.el_deg
    assert abs(look0.az_deg) < 0.05, look0.az_deg
    assert times.along_track_s > 40.0
    assert times.along_track_s < 90.0
    assert times.slant_start_km <= SLANT_MAX_KM + 20.0
    assert times.gsd_band_along_start_m <= GSD_MAX_BAND_M + 5.0
    assert times.peak_el_rate_deg_s < MAX_HW_SLEW_DEG_S
    assert times.az_max_deg < 0.05
    assert optics.fov_az_deg < optics.fov_az_raw_deg
    assert abs(optics.computed_2_3_hfov_deg - OPTICS_SPEC.datasheet_hfov_2_3_deg) < 0.02
    t_eq, _ = origin_window(orbit, optics, box, 0.0, 0.0)
    assert t_eq.az_max_deg > 2.0
    print("self-check: ok")


def _write_geometry_csv(result: GeometryResult, out_dir: Path) -> None:
    """Write design-pass CSV tables.

    Args:
        result: Computed tables.
        out_dir: Output directory.

    Returns:
        None.
    """
    table = result.table
    with (out_dir / "tracking_time_vs_radius.csv").open("w", encoding="utf-8") as fh:
        fh.write("radius_km,az_peak_deg,one_axis_s,two_axis_s,lost_s\n")
        for i, radius in enumerate(table.radius_km):
            fh.write(
                f"{radius:.3f},{table.az_peak_deg[i]:.4f},"
                f"{table.one_axis_s[i]:.3f},{table.two_axis_s[i]:.3f},"
                f"{table.lost_s[i]:.3f}\n"
            )
    with (out_dir / "origin_offset.csv").open("w", encoding="utf-8") as fh:
        fh.write("origin_cross_km,one_axis_s,two_axis_s,lost_s\n")
        for row in result.offsets:
            lost = row.two_axis_s - row.one_axis_s
            fh.write(
                f"{row.origin_cross_km:.1f},{row.one_axis_s:.3f},{row.two_axis_s:.3f},{lost:.3f}\n"
            )
    with (out_dir / "latitude.csv").open("w", encoding="utf-8") as fh:
        fh.write("lat_deg,h_km,el_window_s,az_max_deg,one_axis_R0_s,two_axis_s\n")
        for r0 in result.lat_r0:
            fh.write(
                f"{r0.lat_deg:.4f},{r0.alt_km:.3f},{r0.el_window_s:.3f},"
                f"{r0.az_max_deg:.4f},{r0.one_axis_s:.3f},{r0.two_axis_s:.3f}\n"
            )


def run_geometry() -> GeometryResult:
    """Run the design-pass study: tables, figures, RESULTS.md.

    Returns:
        Computed GeometryResult. Writes under STUDY_DIR / outputs and RESULTS.md.
    """
    from analysis.studies.single_axis_vs_dual_axis_gimbal.figures import (
        plot_along_track,
        plot_disk_angle,
        plot_footprint,
        plot_latitude,
        plot_lost_time,
        plot_offset_map,
        plot_required_az,
        plot_time_vs_radius,
    )
    from analysis.studies.single_axis_vs_dual_axis_gimbal.report import write_geometry_report

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_plot_style()
    optics = build_optics(OPTICS_SPEC)
    orbit = build_orbit(TLE, use_perigee=False)
    orbit_p = build_orbit(TLE, use_perigee=True)
    times, data = origin_window(orbit, optics, GIMBAL_BOX, DESIGN_LAT_DEG, 0.0)
    times_p, _ = origin_window(orbit_p, optics, GIMBAL_BOX, DESIGN_LAT_DEG, 0.0)
    self_check(orbit, optics, GIMBAL_BOX, times)

    table = tracking_time_vs_radius(orbit, optics, GIMBAL_BOX, DESIGN_LAT_DEG, DISK_RADII_KM)
    offsets = offset_times(orbit, optics, GIMBAL_BOX, DESIGN_LAT_DEG)
    lat0 = latitude_table(orbit, optics, GIMBAL_BOX, PASS_LATS_DEG, 0.0)
    omega_rel = omega_rel_max_deg_s(optics.ifov_band_deg)
    omega_img = omega_img_rewind_deg_s(optics.ifov_band_deg, times.peak_el_rate_deg_s)
    hunt = HuntModel(
        optics=optics,
        orbit=orbit,
        box=GIMBAL_BOX,
        omega_img_deg_s=omega_img,
        omega_rel_deg_s=omega_rel,
    ).wait(
        DESIGN_LAT_DEG,
        1.0e-6,
        t_dwell_1_s=times.along_track_s,
        t_dwell_2_s=times.along_track_s,
        t_window_s=times.along_track_s,
    )
    result = GeometryResult(
        orbit=orbit,
        optics=optics,
        box=GIMBAL_BOX,
        times=times,
        origin_samples=data,
        times_perigee=times_p,
        table=table,
        offsets=offsets,
        lat_r0=lat0,
        omega_rel_max_deg_s=omega_rel,
        omega_img_rewind_deg_s=omega_img,
        hunt=hunt,
    )

    plot_along_track(result, OUT_DIR / "along_track_timeline.png")
    plot_disk_angle(result, OUT_DIR / "disk_angular_radius.png")
    plot_time_vs_radius(result, OUT_DIR / "tracking_time_vs_radius.png")
    plot_lost_time(result, OUT_DIR / "lost_time_vs_radius.png")
    plot_footprint(result, OUT_DIR / "footprint_vs_time.png")
    plot_offset_map(result, OUT_DIR / "origin_offset.png")
    plot_required_az(result, OUT_DIR / "required_az_vs_radius.png")
    plot_latitude(result, OUT_DIR / "latitude_earth_rotation.png")
    _write_geometry_csv(result, OUT_DIR)
    write_geometry_report(result, STUDY_DIR / "RESULTS.md")

    print()
    print(f"TLE epoch         {TLE.epoch}")
    print(f"design lat        {DESIGN_LAT_DEG:.2f} deg")
    print(f"local altitude    {times.local_alt_km:.2f} km")
    print(f"usable FOV        {optics.fov_az_deg:.3f} x {optics.fov_el_deg:.3f} deg")
    print(f"along-track time  {times.along_track_s:.1f} s")
    print(f"origin |az| max   {times.az_max_deg:.3f} deg")
    print(f"stare             {times.stare_s:.2f} s")
    print(f"peak el rate      {times.peak_el_rate_deg_s:.3f} deg/s")
    print(f"imaging rewind    {omega_img:.3f} deg/s (gate {omega_rel:.3f} - track)")
    print(f"hardware slew cap {MAX_HW_SLEW_DEG_S:.1f} deg/s")
    half_swath = times.local_alt_km * math.tan(math.radians(optics.half_az_deg))
    print(f"nadir half-swath  {half_swath:.2f} km")
    print(f"wrote {OUT_DIR}")
    return result
