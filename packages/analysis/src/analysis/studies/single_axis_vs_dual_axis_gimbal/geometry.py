"""Design-pass tracking-time tables wrapping analysis.lib.

Computes the one-sided elevation window, Earth-rotation az walk, and
one-axis vs two-axis off-track plume-seconds. On-track dwell still uses
the cluster covering disk. Off-track information uses each plume's L-disk:
loss is when the innermost plume fails the half-disk-on-chip test.
``run_geometry`` writes CSV, figures, and RESULTS.md.

Contains:
  - WindowTimes / LatitudeRow / OffsetRow / GeometryResult.
  - origin_window / latitude_table / cluster_stack_offsets_km / offset_times.
  - self_check / run_geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.lib.hunt import HuntModel, HuntResult
from analysis.lib.look import GimbalBox, look_at
from analysis.lib.optics import Optics, band_gsd_along_m, build_optics
from analysis.lib.orbit import Orbit, argument_of_latitude, build_orbit, iss_eci, origin_ecef
from analysis.lib.plot_style import apply as apply_plot_style
from analysis.lib.tracking import (
    PassSamples,
    SampleSpan,
    disk_half_in_chip,
    in_science_window,
    mask_time_s,
    sample_pass,
    staring_in_frame,
)
from analysis.studies.single_axis_vs_dual_axis_gimbal.assumptions import (
    DESIGN_LAT_DEG,
    GEOMETRY_DT_S,
    GIMBAL_BOX,
    MAX_HW_SLEW_DEG_S,
    OFFSET_PLANT_D_KM,
    OFFSET_STACK_N,
    OPTICS_SPEC,
    ORIGIN_OFFSETS_KM,
    OUT_DIR,
    PASS_LATS_DEG,
    PLUME_R_KM,
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
    """One cross-track cluster-centroid offset at the design pass.

    Plume-seconds sum in-frame time over the representative stacks. A
    plume counts when at least half its L-disk is on the chip. The
    in-frame count is how many plumes have any in-frame time; the
    one-sided window ends at nadir, so a closest-approach snapshot
    is not a valid in-window sample.

    Attributes:
        origin_cross_km: Cluster-centroid cross-track offset, kilometres.
        one_axis_plume_s: One-axis plume-seconds (parked az = 0).
        two_axis_plume_s: Two-axis plume-seconds (boresight each plume in box).
        one_axis_n_in: One-axis plumes with any in-frame time.
        two_axis_n_in: Two-axis plumes with any in-frame time.
    """

    origin_cross_km: float
    one_axis_plume_s: float
    two_axis_plume_s: float
    one_axis_n_in: int
    two_axis_n_in: int


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
        offsets: Cross-track cluster offsets at the design lat.
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
    el_mask = in_science_window(data, box)
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
    rows: list[LatitudeRow] = []
    for lat in lats:
        times, origin = origin_window(orbit, optics, box, lat, 0.0)
        science = in_science_window(origin, box)
        one = science & disk_half_in_chip(
            origin.az, origin.slant, radius_km, optics, boresight_origin=False
        )
        two = science & disk_half_in_chip(
            origin.az,
            origin.slant,
            radius_km,
            optics,
            boresight_origin=True,
            az_box_deg=box.az_box_deg,
        )
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


def cluster_stack_offsets_km(n: int, d_km: float) -> tuple[float, ...]:
    """Return cross-track stack offsets for a plant of covering radius ``d_km``.

    Stacks lie on a cross-track line from -D to +D. n = 1 is a singleton
    at the centroid. Along-track spread does not change the parked-az story.

    Args:
        n: Stack count. Must be >= 1.
        d_km: Plant-span covering radius in kilometres.

    Returns:
        Cross-track offsets in kilometres, length n.

    Raises:
        ValueError: If ``n`` is less than 1.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return (0.0,)
    return tuple(float(x) for x in np.linspace(-d_km, d_km, n))


def offset_times(
    orbit: Orbit,
    optics: Optics,
    box: GimbalBox,
    lat_deg: float,
) -> list[OffsetRow]:
    """Return per-plume information vs cluster-centroid cross-track offset.

    Each stack has its own L-disk. One-axis parks at az = 0. Two-axis
    boresights that plume when it is inside the keep-out box. Information
    is the sum of in-frame times (plume-seconds). A cluster is fully lost
    when even the innermost plume fails the half-disk test.

    Args:
        orbit: Circular ISS orbit.
        optics: Usable sensor FOV.
        box: Gimbal box.
        lat_deg: Pass latitude in degrees.

    Returns:
        One row per offset in ORIGIN_OFFSETS_KM.
    """
    dys = cluster_stack_offsets_km(OFFSET_STACK_N, OFFSET_PLANT_D_KM)
    span = _span()
    rows: list[OffsetRow] = []
    for y_km in ORIGIN_OFFSETS_KM:
        plume_s_1 = 0.0
        plume_s_2 = 0.0
        n_in_1 = 0
        n_in_2 = 0
        for dy in dys:
            data = sample_pass(orbit, box, lat_deg, 0.0, float(y_km + dy), span)
            science = in_science_window(data, box)
            one = science & disk_half_in_chip(
                data.az, data.slant, PLUME_R_KM, optics, boresight_origin=False
            )
            two = science & disk_half_in_chip(
                data.az,
                data.slant,
                PLUME_R_KM,
                optics,
                boresight_origin=True,
                az_box_deg=box.az_box_deg,
            )
            t1 = mask_time_s(one, data.t)
            t2 = mask_time_s(two, data.t)
            plume_s_1 += t1
            plume_s_2 += t2
            if t1 > 0.0:
                n_in_1 += 1
            if t2 > 0.0:
                n_in_2 += 1
        rows.append(
            OffsetRow(
                origin_cross_km=float(y_km),
                one_axis_plume_s=plume_s_1,
                two_axis_plume_s=plume_s_2,
                one_axis_n_in=n_in_1,
                two_axis_n_in=n_in_2,
            )
        )
    return rows


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
    with (out_dir / "origin_offset.csv").open("w", encoding="utf-8") as fh:
        fh.write("origin_cross_km,one_axis_plume_s,two_axis_plume_s,one_axis_n_in,two_axis_n_in\n")
        for row in result.offsets:
            fh.write(
                f"{row.origin_cross_km:.1f},{row.one_axis_plume_s:.3f},"
                f"{row.two_axis_plume_s:.3f},{row.one_axis_n_in},"
                f"{row.two_axis_n_in}\n"
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
        plot_az_walk,
        plot_offset_map,
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
        offsets=offsets,
        lat_r0=lat0,
        omega_rel_max_deg_s=omega_rel,
        omega_img_rewind_deg_s=omega_img,
        hunt=hunt,
    )

    plot_along_track(result, OUT_DIR / "along_track_timeline.png")
    plot_az_walk(result, OUT_DIR / "az_walk_vs_time.png")
    plot_offset_map(result, OUT_DIR / "origin_offset.png")
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
