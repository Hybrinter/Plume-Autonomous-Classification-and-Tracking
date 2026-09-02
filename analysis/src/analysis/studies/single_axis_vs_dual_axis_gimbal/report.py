"""Markdown reports for the single-axis vs dual-axis gimbal study.

``write_geometry_report`` writes RESULTS.md (design-pass geometry).
``write_industry_report`` writes INDUSTRIAL.md (world stack inventory).
``write_study_readme`` writes the study folder README.md.
All three are generated artifacts, not STE descriptive docs.

Contains:
  - write_geometry_report / write_industry_report / write_study_readme.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from analysis.lib.optics import Optics, build_optics
from analysis.lib.orbit import Orbit, build_orbit
from analysis.lib.tracking import TimeLostFn
from analysis.studies.single_axis_vs_dual_axis_gimbal.assumptions import (
    BAND_ALONG_PX,
    BAND_LATERAL_PX,
    CAMERA_NAME,
    CLS_FLOPS_256_G,
    CLS_LAT_256_MS,
    DESIGN_LAT_DEG,
    EXPOSURE_S,
    GIMBAL_BOX,
    MAX_HW_SLEW_DEG_S,
    MAX_SMEAR_BAND_PX,
    OPTICS_SPEC,
    PLUME_R_KM,
    SEG_FLOPS_256_G,
    SEG_LAT_256_MS,
    SENSOR_NAME,
    TLE,
    WIND_MPS,
)
from analysis.studies.single_axis_vs_dual_axis_gimbal.inventory import Cluster
from analysis.studies.single_axis_vs_dual_axis_gimbal.profile import (
    LatBandRow,
    RadiusProfile,
    encounter_time_s,
    recovered_s,
)

if TYPE_CHECKING:
    from analysis.studies.single_axis_vs_dual_axis_gimbal.geometry import (
        GeometryResult,
        WindowTimes,
    )

_RUN_CMD = "uv run python -m analysis.studies.single_axis_vs_dual_axis_gimbal"


def _setup_lines(*, include_heading: bool) -> list[str]:
    """Return the physical-setup bullets shared by generated reports.

    Args:
        include_heading: If True, include the section heading.

    Returns:
        Markdown lines without a trailing blank.
    """
    lines: list[str] = []
    if include_heading:
        lines.append("## Physical setup")
        lines.append("")
    lines.extend(
        [
            "- **Fixed origin.** Elevation tracks the stack, or the cluster centroid of",
            "  stacks. Wind can swing the visible plume around that point during the",
            "  ~120 s window. Tracking plume CoG would add lateral walk this study",
            "  does not model.",
            "- **Covering disk.** Unknown wind azimuth makes a disk of radius **L**",
            "  around each stack. Cluster covering radius is `R = D + L`, with D the",
            "  haversine plant span. That is a possibility set, not a smoke-filled blob.",
            f"- **L = {PLUME_R_KM:.0f} km** is a **conservative visible-length envelope**",
            "  (cooling-tower photo climatologies: typical 0.3-0.8 km, winter often",
            "  >0.9 km, ~90-95th percentile ~1.5-3 km). Not a Mommert-chip measurement.",
            "  Not typical length.",
            "- **P(visible)** = fraction of that disk in the FOV (unknown-wind geometry).",
            "  Plume volume / Gaussian ribbon (~5% disk fill) is occupancy/SNR only.",
            "  Do not multiply the two.",
            "- **No locked design R.** Operational R is per cluster (`r_cover_km`) and,",
            "  for lat tables, stack-weighted mean `R(|lat|) = D(|lat|) + L`.",
            f"- **Camera/lens.** {CAMERA_NAME}, {SENSOR_NAME}, 3.45 um, 2448x2048,",
            "  2/3-inch, global shutter; 150 mm athermal, 0.66% distortion. Catalog",
            "  2/3-inch HFOV 3.36 deg is a check. 2x2 mosaic => band plane",
            f"  {BAND_LATERAL_PX}x{BAND_ALONG_PX}, IFOV x2. Ignore lens-catalog 2.74 um.",
            "- **Two slew caps.** Imaging rewind vs ground is the 1-pixel / 1 ms",
            "  **band-plane** gate (current imaging gate). Hardware cap",
            f"  **{MAX_HW_SLEW_DEG_S:.0f} deg/s** is not for science frames. SIL",
            "  `ifov=0.02` is not this optic.",
            "- **Reacquire** starts when leftover window opens (hunt during slew).",
            "  Do not wait until elevation is back at 30 deg. One-axis hunt is",
            "  elevation-only (FOV ribbon). Two-axis hunt would use the gimbal box.",
            "  Mean encounter is Poisson / mean-spacing, ocean-averaged in a lat band,",
            "  conditional on the ISS belt -- not a city-corridor nearest neighbour.",
        ]
    )
    return lines


def _band_times(
    fn: TimeLostFn,
    row: LatBandRow,
    optics: Optics,
    omega_img: float,
    orbit: Orbit,
) -> tuple[float, float, float, float, float]:
    """Return T1, T2, T1_eff, ribbon encounter, box encounter for one lat band.

    Args:
        fn: Tracking-time interpolator.
        row: Folded |lat| stats, including mean R and cluster density.
        optics: Usable sensor FOV.
        omega_img: Imaging rewind cap in degrees per second.
        orbit: Circular ISS orbit.

    Returns:
        (t1, t2, t1_eff, t_enc_ribbon, t_enc_box) in seconds.
    """
    t1, t2, _lost = fn.eval(row.lat_mid, row.mean_r_km)
    leftover = t2 - t1
    h_km = orbit.local_altitude_km(row.lat_mid)
    swath1 = 2.0 * h_km * math.tan(math.radians(optics.half_az_deg))
    swath2 = 2.0 * h_km * math.tan(math.radians(GIMBAL_BOX.az_box_deg))
    v_scan = h_km * math.radians(omega_img)
    t_enc1 = encounter_time_s(row.dens_per_km2, swath1, v_scan)
    t_enc2 = encounter_time_s(row.dens_per_km2, swath2, v_scan)
    t1_eff = min(t2, t1 + recovered_s(leftover, t_enc1))
    return t1, t2, t1_eff, t_enc1, t_enc2


def _fmt_enc(t_s: float) -> str:
    """Return a seconds string for a mean encounter time.

    Args:
        t_s: Encounter time in seconds, possibly inf.

    Returns:
        Formatted seconds, or 'inf' if not finite.
    """
    if not math.isfinite(t_s):
        return "inf"
    if t_s >= 1000.0:
        return f"{t_s:.0f}"
    return f"{t_s:.1f}"


def write_geometry_report(result: GeometryResult, path: Path) -> None:
    """Write RESULTS.md for the design-pass geometry.

    Args:
        result: Design-pass tables.
        path: Output markdown path.

    Returns:
        None.
    """
    orbit = result.orbit
    optics = result.optics
    times = result.times
    table = result.table
    times_perigee = result.times_perigee
    h_km = times.local_alt_km
    r_fit = h_km * math.tan(math.radians(optics.half_az_deg))
    r_box = h_km * math.tan(math.radians(GIMBAL_BOX.az_box_deg))
    area_256 = 256 * 256
    area_ff = BAND_LATERAL_PX * BAND_ALONG_PX
    scale = area_ff / area_256
    omega_rel = result.omega_rel_max_deg_s
    omega_img = result.omega_img_rewind_deg_s
    lines: list[str] = []
    a = lines.append
    a("# Single-axis gimbal tracking time")
    a("")
    a("TEMPORARY ANALYSIS. Generated by the geometry command of")
    a(f"`{_RUN_CMD}`. Not flight software.")
    a("")
    lines.extend(_setup_lines(include_heading=True))
    a("")
    a("Also locked for this run:")
    a("")
    a("- Elevation window is **one-sided** 90->30 deg. Hard gimbal stops for")
    a("  clearance; tracking time stops at the limit, with no leftover FOV walk-out.")
    a("- Mount is geocentric nadir. A <=5 deg offset is a later correction, not in")
    a("  this run.")
    a("- If plants are spread farther than the chip (~+/-12 km at nadir), they are")
    a("  two clusters. Do not average across a city.")
    a("- Current ISS TLE (Celestrak GP, epoch 2026-09-01). Earth rotation and")
    a("  51.63 deg inclination are on. Design pass is **max latitude** (heading")
    a("  due east), which is the polar-most nadir ISS can reach.")
    a("- Full-frame inference = the full 2x2 band plane. That does not change")
    a("  optical FOV; it is a compute note.")
    a("- A ~2028 launch still sits in the current 416-423 km operational band.")
    a("  ISS is committed through 2030; retrograde deorbit lowering is a")
    a("  2028-2030 lead-up, not a 2028 operations baseline.")
    a("")
    a("## ISS orbit (TLE 25544, 2026-09-01)")
    a("")
    a("| Quantity | Value |")
    a("| --- | --- |")
    a(f"| Inclination | {TLE.inclination_deg:.4f} deg |")
    a(f"| Eccentricity | {TLE.eccentricity:.7f} |")
    a(f"| Mean motion | {TLE.mean_motion_rev_per_day:.8f} rev/day |")
    a(f"| Period | {orbit.period_s / 60.0:.3f} min |")
    a(f"| SMA | {orbit.sma_km:.2f} km |")
    a(f"| Mean altitude (WGS84 equator) | {orbit.mean_alt_eq_km:.2f} km |")
    a(
        f"| Perigee / apogee altitude | "
        f"{orbit.perigee_alt_eq_km:.2f} / {orbit.apogee_alt_eq_km:.2f} km |"
    )
    a(f"| Inertial speed (SMA) | {orbit.v_km_s:.3f} km/s |")
    a(f"| Local altitude at lat {DESIGN_LAT_DEG:.2f} deg | **{h_km:.2f} km** |")
    a(f"| Local altitude at perigee, same lat | {times_perigee.local_alt_km:.2f} km |")
    a("")
    a("A ~2028 launch does not change this band. NASA holds ~415 km through")
    a("2030; debris risk rises if the station is boosted higher. Perigee in this")
    a(f"TLE is {orbit.perigee_alt_eq_km:.1f} km equatorial; the one-sided window")
    a(f"there is {times_perigee.along_track_s:.1f} s vs {times.along_track_s:.1f} s at SMA.")
    a("")
    a("## Optics (most restrictive)")
    a("")
    a("| Quantity | Value |")
    a("| --- | --- |")
    a(f"| Camera | {CAMERA_NAME} |")
    a(f"| Sensor | {SENSOR_NAME} |")
    a(f"| Array | {OPTICS_SPEC.n_lateral_px} x {OPTICS_SPEC.n_along_px} (lateral x along-track) |")
    a(f"| Pixel pitch | {OPTICS_SPEC.pixel_um:.2f} um (not the lens-catalog 2.74 um) |")
    a(f"| Active area | {optics.sensor_width_mm:.3f} x {optics.sensor_height_mm:.3f} mm |")
    a(f"| Thin-lens FOV | {optics.fov_az_raw_deg:.3f} x {optics.fov_el_raw_deg:.3f} deg |")
    a(
        f"| Usable FOV after {OPTICS_SPEC.lens_distortion_pct:.2f} % distortion | "
        f"**{optics.fov_az_deg:.3f} x {optics.fov_el_deg:.3f} deg** |"
    )
    a(f"| Half-FOV | +/-{optics.half_az_deg:.3f} deg az, +/-{optics.half_el_deg:.3f} deg el |")
    a(
        f"| 2/3-inch sheet HFOV (8.8 mm, not used) | {optics.datasheet_hfov_deg:.2f} deg "
        f"(computed {optics.computed_2_3_hfov_deg:.3f}) |"
    )
    a(f"| Mosaic IFOV | {optics.ifov_deg:.5f} deg/px |")
    a(f"| Band-plane IFOV (2x2) | {optics.ifov_band_deg:.5f} deg/px |")
    gsd = h_km * 1000.0 / (OPTICS_SPEC.focal_length_mm / 1000.0) * OPTICS_SPEC.pixel_um * 1e-6
    a(f"| Nadir GSD at design lat (mosaic pixel) | **{gsd:.2f} m** |")
    a(f"| Nadir GSD (2x2 band cell) | {2 * gsd:.2f} m |")
    a(f"| Nadir lateral half-swath | **+/-{r_fit:.2f} km** |")
    a(f"| 2-axis +/-{GIMBAL_BOX.az_box_deg:.0f} deg half-swath | {r_box:.1f} km |")
    a("")
    a("## Imaging gate and slew caps (current imaging gate)")
    a("")
    a("Science frames use a 1 ms exposure and a 1 **band-plane** pixel smear")
    a("budget. Scene-relative rate on rewind is gimbal slew plus ISS track,")
    a("so the imaging rewind cap is the smear limit minus peak elevation rate.")
    a(f"The hardware cap of {MAX_HW_SLEW_DEG_S:.0f} deg/s is a motor limit, not")
    a("a science-frame limit. This gate is the current imaging assumption;")
    a("a shorter exposure or a looser smear budget would raise it.")
    a("")
    a("| Quantity | Value |")
    a("| --- | --- |")
    a(f"| Exposure | {EXPOSURE_S * 1e3:.1f} ms |")
    a(f"| Smear budget | {MAX_SMEAR_BAND_PX:.0f} band-plane pixel |")
    a(f"| omega_rel_max = IFOV_band / exposure | **{omega_rel:.3f} deg/s** |")
    a(f"| Peak ISS elevation rate (design pass) | {times.peak_el_rate_deg_s:.3f} deg/s |")
    a(f"| Imaging rewind omega_img | **{omega_img:.3f} deg/s** |")
    a(f"| Hardware slew cap | {MAX_HW_SLEW_DEG_S:.1f} deg/s |")
    a("")
    a("## Along-track time (elevation axis, design pass)")
    a("")
    a(f"Design pass: latitude **{DESIGN_LAT_DEG:.2f} deg**, heading due east,")
    a("Earth rotation almost purely along-track.")
    a("")
    a("| Quantity | Value |")
    a("| --- | --- |")
    a("| Elevation window | 90 -> 30 deg, one-sided, stop at limit |")
    a(f"| Time in elevation window | **{times.along_track_s:.1f} s** |")
    a(f"| Staring at nadir (no gimbal) | {times.stare_s:.2f} s |")
    a(f"| Peak elevation rate | {times.peak_el_rate_deg_s:.3f} deg/s |")
    a(f"| Peak |az| of the origin (Earth rotation) | {times.az_max_deg:.3f} deg |")
    a(f"| Window start / stop from CA | {times.t_start_s:.1f} / {times.t_stop_s:.1f} s |")
    a(
        f"| Slant range at start / stop | "
        f"{times.slant_start_km:.1f} / {times.slant_stop_km:.1f} km |"
    )
    a("")
    a("## Earth rotation vs pass latitude")
    a("")
    a("At low latitude the ISS heading has a large north component, so Earth")
    a("rotation (always east) has a **cross-track** piece. That walks even a")
    a("nadir-centered origin in azimuth, peaking at the 30 deg stop where range")
    a("is long. At max latitude the heading is due east and the walk vanishes.")
    a("That is why a polar-slice one-axis placement is the right one-axis case.")
    a("")
    a("This table is a **point origin (R = 0)**. Operational covering radius")
    a("is `R(|lat|) = D(|lat|) + L` from the Climate TRACE plant span; that")
    a("profile and the T(lat, R(|lat|)) curves are in `INDUSTRIAL.md`.")
    a("")
    a("| lat (deg) | h (km) | el window (s) | peak az origin (deg) | 1-axis R=0 (s) | 2-axis (s) |")
    a("| --- | --- | --- | --- | --- | --- |")
    for r0 in result.lat_r0:
        a(
            f"| {r0.lat_deg:.2f} | {r0.alt_km:.1f} | {r0.el_window_s:.1f} | "
            f"{r0.az_max_deg:.3f} | {r0.one_axis_s:.1f} | {r0.two_axis_s:.1f} |"
        )
    a("")
    a("For a point origin, Earth rotation walks it out of the chip below ~40 deg")
    a("latitude. Below that, one-axis only keeps the last tens of seconds near")
    a("nadir. How much a real cluster loses depends on R(|lat|), not on a")
    a("locked design radius.")
    a("")
    a("## Covering-disk sensitivity (design pass)")
    a("")
    a("Elevation tracks the fixed stack / cluster centroid. A plume is acquired")
    a("if any part of the unknown-wind disk is in the frame. The radii below")
    a("are a **sensitivity sweep at the design latitude**, not a locked")
    a(f"operational R. Isolated stacks sit at R = L = {PLUME_R_KM:.0f} km.")
    a("")
    a("| R (km) | peak az (deg) | 1-axis (s) | 2-axis (s) | lost (s) | lost % |")
    a("| --- | --- | --- | --- | --- | --- |")
    for i, radius in enumerate(table.radius_km):
        t1 = float(table.one_axis_s[i])
        t2 = float(table.two_axis_s[i])
        lost = float(table.lost_s[i])
        frac = 0.0 if t2 <= 0 else 100.0 * lost / t2
        a(
            f"| {radius:.1f} | {table.az_peak_deg[i]:.2f} | {t1:.1f} | {t2:.1f} | "
            f"{lost:.1f} | {frac:.1f}% |"
        )
    a("")
    a(f"At the design pass, a disk with **R <= {r_fit:.1f} km** stays in the chip")
    a("for the whole elevation window. That includes every operational cluster")
    a(f"in this inventory (R = D + {PLUME_R_KM:.0f} km, D capped by the chip split).")
    a("")
    a("Wind during the window (extra walk of a visible plume around the fixed")
    a("origin; already inside the L envelope if L is conservative):")
    a("")
    a("| wind (m/s) | extra displacement (km) | leftover 1-axis budget (km) |")
    a("| --- | --- | --- |")
    for wind in WIND_MPS:
        extra = (wind / 1000.0) * times.along_track_s
        a(f"| {wind:.0f} | {extra:.2f} | {r_fit - extra:.2f} |")
    a("")
    a("## Off-center cluster (design pass, R = 0)")
    a("")
    a("| origin cross-track (km) | 1-axis (s) | 2-axis (s) | lost (s) |")
    a("| --- | --- | --- | --- |")
    for row in result.offsets:
        a(
            f"| {row.origin_cross_km:.0f} | {row.one_axis_s:.1f} | "
            f"{row.two_axis_s:.1f} | {row.two_axis_s - row.one_axis_s:.1f} |"
        )
    a("")
    a("## Full-frame inference")
    a("")
    a("Optical FOV is the full chip either way. Full-frame inference changes")
    a("compute, not tracking time. The 2x2 band plane is")
    a(f"**{BAND_LATERAL_PX} x {BAND_ALONG_PX}**.")
    a("")
    a("| | 256x256 (today) | full band plane | scale |")
    a("| --- | --- | --- | --- |")
    a(f"| Pixels | {area_256} | {area_ff} | {scale:.1f}x |")
    a(
        f"| DilateNet-w32 FLOPs | {SEG_FLOPS_256_G:.2f} G | "
        f"{SEG_FLOPS_256_G * scale:.2f} G | {scale:.1f}x |"
    )
    a(
        f"| ShuffleNet FLOPs | {CLS_FLOPS_256_G:.3f} G | "
        f"{CLS_FLOPS_256_G * scale:.2f} G | {scale:.1f}x |"
    )
    a(
        f"| CPU accept latency (linear scale from 256) | "
        f"{SEG_LAT_256_MS:.1f} / {CLS_LAT_256_MS:.1f} ms | "
        f"{SEG_LAT_256_MS * scale:.0f} / {CLS_LAT_256_MS * scale:.0f} ms | |"
    )
    a("")
    a("The 500x parameter cut (DilateNet-w32 23.5 k vs U-Net 13.4 M) is what")
    a("makes full-frame plausible. FLOPs still fit the 500 ms budget if latency")
    a("scales with pixels. Jetson Nano (4 GB) is tighter on **activation memory**")
    a("at 1224x1024 than Orin NX; that needs a bench, not a FLOP argument.")
    a("A 256-pixel centre crop on this optic is only ~0.67 deg -- that would")
    a("throw away the 1-axis lateral budget. Full-frame is the right inference")
    a("size for this lens.")
    a("")
    a("## Bottom line")
    a("")
    a(f"1. Along-track, one elevation axis gives **{times.along_track_s:.0f} s**")
    a(f"   at the design (max-lat) pass, vs **{times.stare_s:.1f} s** staring.")
    a(f"   Stop at 30 deg. Peak track rate {times.peak_el_rate_deg_s:.2f} deg/s")
    a(f"   is under the {MAX_HW_SLEW_DEG_S:.0f} deg/s hardware cap. Imaging")
    a(f"   rewind against the pass is capped at **{omega_img:.2f} deg/s**.")
    a("2. At that pass, Earth-rotation az walk of the origin is")
    a(f"   **{times.az_max_deg:.3f} deg**. Any disk with R <= {r_fit:.1f} km")
    a(f"   stays in the +/-{optics.half_az_deg:.2f} deg chip for the whole")
    a("   window -- **0 s lost** vs two-axis. There is no locked design R;")
    a("   the inventory R(|lat|) is always below that chip radius.")
    a("3. At equatorial and mid-latitude passes, Earth rotation walks the")
    a("   origin out at the 30 deg stop. One-axis then keeps only the last")
    a("   tens of seconds, worse for larger R(|lat|). Use two-axis for those")
    a("   passes, or do not work them. See `INDUSTRIAL.md` for T vs R(|lat|).")
    a("4. If two plants are farther apart than ~24 km, they are two clusters.")
    a("   Track one. Leftover one-axis time hunts immediately at omega_img.")
    a("5. Worldwide stack inventory (Climate TRACE 2025) is in")
    a("   `INDUSTRIAL.md`.")
    a("")
    a("## Figures")
    a("")
    a("- `outputs/along_track_timeline.png`")
    a("- `outputs/disk_angular_radius.png`")
    a("- `outputs/tracking_time_vs_radius.png`")
    a("- `outputs/lost_time_vs_radius.png`")
    a("- `outputs/footprint_vs_time.png`")
    a("- `outputs/origin_offset.png`")
    a("- `outputs/required_az_vs_radius.png`")
    a("- `outputs/latitude_earth_rotation.png`")
    a("- `outputs/tracking_time_vs_radius.csv`, `outputs/origin_offset.csv`,")
    a("  `outputs/latitude.csv`")
    a("")
    a("```text")
    a(f"{_RUN_CMD} geometry")
    a("```")
    a("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_industry_report(
    clusters: list[Cluster],
    gem_n: int,
    gppd_n: int,
    ct_n: int,
    fn: TimeLostFn,
    exp: dict[str, dict[str, float]],
    profile: RadiusProfile,
    folded: list[LatBandRow],
    omega_img: float,
    path: Path,
) -> None:
    """Write INDUSTRIAL.md for the worldwide stack inventory.

    Args:
        clusters: World plant clusters.
        gem_n: GEM operating source count (cross-check).
        gppd_n: GPPD thermal plant count (cross-check).
        ct_n: Climate TRACE source count.
        fn: Tracking-time interpolator.
        exp: Expected-time summaries keyed by weight kind.
        profile: Stack-weighted R(|lat|) interpolator.
        folded: Coarse |lat| band rows (D, R, singleton fraction).
        omega_img: Imaging rewind cap in degrees per second.
        path: Output markdown path.

    Returns:
        None.
    """
    optics = build_optics(OPTICS_SPEC)
    orbit = build_orbit(TLE, use_perigee=False)
    iss_i = TLE.inclination_deg
    world = clusters
    iss = [c for c in world if abs(c.lat) <= iss_i]
    lines: list[str] = []
    p = lines.append
    p("# Industrial area vs latitude, and expected tracking time")
    p("")
    p("TEMPORARY ANALYSIS. Generated by the industry command of")
    p(f"`{_RUN_CMD}`. Not flight software.")
    p("")
    lines.extend(_setup_lines(include_heading=True))
    p("")
    p("## Inventory")
    p("")
    p("Primary: Climate TRACE v5.10.0 CO2 **point sources** (2025, monthly summed)")
    p("that actually have smoke stacks:")
    p("")
    p("- combustion electricity generation (coal / gas / oil / biomass / waste)")
    p("- cement, lime, glass")
    p("- iron and steel, aluminum, other metals")
    p("- chemicals, other chemicals, petrochemical steam cracking")
    p("- pulp and paper, other manufacturing")
    p("- oil and gas refining")
    p("")
    p("Oil/gas production, mines, hydro/solar/wind, and food plants are out:")
    p("they are not the industrial-stack plume the payload is built for.")
    p("")
    p("Nearby sources within 8 km are one plant cluster. Clusters wider than")
    p("the chip (~12 km covering radius) are split, matching the earlier rule")
    p("that two plants farther apart than the chip are two tracks. Covering")
    p(f"radius is `R = D + {PLUME_R_KM:.0f} km` (plant span + L envelope).")
    p("")
    p("The size assumption is that industrial-area size scales with the")
    p("number of stacks, so the **primary weight is stack count**. Covering-disk")
    p("area over-weights sprawling clusters (pi D^2) and is a sensitivity only.")
    p("Emissions weight how much the plant is actually running.")
    p("")
    p("| Inventory | sources |")
    p("| --- | --- |")
    p(f"| Climate TRACE stack sources | {ct_n} |")
    p(f"| Clusters (after split) | {len(world)} |")
    p(f"| In ISS belt |lat| <= {iss_i:.2f} deg | {len(iss)} |")
    p(f"| GEM operating (cross-check) | {gem_n} |")
    p(f"| GPPD thermal (cross-check) | {gppd_n} |")
    p("")
    p("GEM and GPPD peak in the same 30-40 N band, with ~93% of plants")
    p("inside the ISS belt. The Climate TRACE latitude shape is not an")
    p("OpenStreetMap-Europe artefact.")
    p("")
    p("## World distribution")
    p("")
    p("ISS never nadir-overflies outside +/-51.63 deg. The 2-axis +/-10 deg box adds")
    p("~0.7 deg of latitude at the turning point; ignored here.")
    p("")
    world_area = sum(c.area_km2 for c in world)
    iss_area = sum(c.area_km2 for c in iss)
    world_n = sum(c.n for c in world)
    iss_n = sum(c.n for c in iss)
    world_em = sum(c.emissions_t for c in world)
    iss_em = sum(c.emissions_t for c in iss)
    p("| | world | ISS belt | ISS fraction |")
    p("| --- | --- | --- | --- |")
    p(
        f"| covering area | {world_area:.0f} km2 | {iss_area:.0f} km2 | "
        f"{100.0 * iss_area / max(1, world_area):.1f}% |"
    )
    p(f"| stack sources | {world_n} | {iss_n} | {100.0 * iss_n / max(1, world_n):.1f}% |")
    p(
        f"| 2025 CO2 | {world_em / 1e9:.2f} Gt | {iss_em / 1e9:.2f} Gt | "
        f"{100.0 * iss_em / max(1, world_em):.1f}% |"
    )
    p("")
    p("Stack-weighted mean covering radius in the ISS belt:")
    p(
        f"**{exp['stacks']['mean_r_km']:.2f} km** "
        f"(mean sources/cluster {exp['stacks']['mean_n']:.2f})."
    )
    p("That is an inventory mean, not a locked design R.")
    p(
        f"Interpolated R(|lat|) at 0 / 30 / {iss_i:.2f} deg: "
        f"{profile.r_km(0.0):.2f} / {profile.r_km(30.0):.2f} / "
        f"{profile.r_km(iss_i):.2f} km."
    )
    p("Stack fraction at |lat| >= 45 deg:")
    p(f"**{100.0 * exp['stacks']['frac_lat_ge_45']:.1f}%**.")
    p("")
    p("### Covering radius vs |latitude|")
    p("")
    p("Stack-weighted plant span D and covering radius R = D + L. Singleton")
    p("fraction is the share of stacks in n=1 clusters. d_char for n>=2 is")
    p("stack-weighted 2 D / sqrt(n), a characteristic neighbour spacing")
    p("inside multi-stack plants. Density is ocean-averaged over the whole")
    p("lat band (ISS belt, both hemispheres), not a city corridor.")
    p("")
    p("| |lat| (deg) | stacks | % of ISS-belt | D (km) | R (km) | singleton % | d_char n>=2 (km) |")
    p("| --- | --- | --- | --- | --- | --- | --- |")
    iss_n_total = max(1, sum(c.n for c in iss))
    for row in folded:
        frac = 100.0 * row.n_stacks / iss_n_total
        p(
            f"| {row.lat_lo:.0f}-{row.lat_hi:.1f} | {row.n_stacks} | {frac:.1f}% | "
            f"{row.mean_d_km:.2f} | {row.mean_r_km:.2f} | "
            f"{100.0 * row.frac_singleton_stacks:.1f}% | {row.d_char_n2_km:.2f} |"
        )
    p("")
    p("Fine 2 deg bins: `outputs/r_vs_lat.csv`.")
    p("")
    p("### Tracking time at R(|lat|), with leftover hunt")
    p("")
    p("T1 / T2 use each band's stack-weighted mean R, not a locked 5 km.")
    p("Leftover = T2 - T1 is front-loaded at low latitude. Hunt starts")
    p("immediately at omega_img (no 30 deg reset). Mean encounter uses")
    p("band cluster density x 1-axis ribbon swath x (h x omega_img).")
    p("T1_eff = min(T2, T1 + max(0, leftover - t_enc)). Two-axis leftover")
    p("hunt would use the +/-10 deg box (~6x wider); 2-axis is still on")
    p("the original origin during leftover, so T1_eff is a 1-axis recovery.")
    p("")
    p(
        "| |lat| (deg) | R (km) | 1-axis (s) | 2-axis (s) | T1_eff (s) | "
        "lost % | lost % reacq | t_enc ribbon (s) | t_enc box (s) |"
    )
    p("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in folded:
        t1, t2, t1_eff, t_enc1, t_enc2 = _band_times(fn, row, optics, omega_img, orbit)
        lost_pct = 0.0 if t2 <= 0 else 100.0 * (t2 - t1) / t2
        lost_eff = 0.0 if t2 <= 0 else 100.0 * (t2 - t1_eff) / t2
        p(
            f"| {row.lat_lo:.0f}-{row.lat_hi:.1f} | {row.mean_r_km:.2f} | "
            f"{t1:.1f} | {t2:.1f} | {t1_eff:.1f} | {lost_pct:.0f}% | "
            f"{lost_eff:.0f}% | {_fmt_enc(t_enc1)} | {_fmt_enc(t_enc2)} |"
        )
    p("")
    p(f"Imaging rewind used here: **{omega_img:.2f} deg/s** (current imaging gate).")
    p("")
    p("## Expected tracking time (ISS-belt clusters, nadir-centered pass)")
    p("")
    p("For each cluster, `time_lost(|lat|, R)` uses that cluster's own")
    p("`r_cover_km` and the same one-sided 90->30 deg window, Earth rotation,")
    p("and hard gimbal stop as the geometry command. A pass is assumed over")
    p("the cluster origin (best-case). Off-track origins are a separate loss,")
    p("captured in the swath column. Reacquire columns hunt during leftover")
    p("at omega_img along the 1-axis ribbon.")
    p("")
    p(
        "| weight | E[T 1-axis] (s) | E[T 2-axis] (s) | E[lost] (s) | lost % | "
        "E[T1_eff] (s) | lost % reacq | weight at |lat|>=45 |"
    )
    p("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for key, label in (
        ("stacks", "**stack count (primary)**"),
        ("emissions", "2025 CO2"),
        ("area", "covering-disk area (sprawl)"),
    ):
        e = exp[key]
        p(
            f"| {label} | {e['e_t1']:.1f} | {e['e_t2']:.1f} | {e['e_lost']:.1f} | "
            f"{e['e_lost_pct']:.1f}% | {e['e_t1_reacq']:.1f} | "
            f"{e['e_lost_reacq_pct']:.1f}% | {100.0 * e['frac_lat_ge_45']:.1f}% |"
        )
    p("")
    e = exp["stacks"]
    p("ISS dwell (more time near +/-i) does not change the story:")
    p(
        f"stacks + dwell -> E[lost] = **{e['e_lost_dwell']:.1f} s** "
        f"({e['e_t1_dwell']:.1f} vs {e['e_t2_dwell']:.1f})."
    )
    p(
        f"Leftover hunt at {omega_img:.2f} deg/s raises stack-weighted T1 from "
        f"{e['e_t1']:.1f} s to **{e['e_t1_reacq']:.1f} s** "
        f"(lost {e['e_lost_reacq']:.1f} s, {e['e_lost_reacq_pct']:.1f}%)."
    )
    p("Ocean-averaged lat-band density makes encounter times long; city")
    p("corridors would reacquire faster. This is not a nearest-neighbour")
    p("plant-to-plant model.")
    p("")
    p("## Lateral swath (why 2-axis still buys something at 50 deg)")
    p("")
    p("Tracking time is conditional on the cluster being in the frame.")
    p("1-axis only sees +/-1.60 deg (~12 km at nadir). 2-axis sees the +/-10 deg")
    p("box (~76 km). Ground-track spacing is thousands of km, so daily")
    p("longitude coverage is proportional to swath.")
    p("")
    p("| | 1-axis | 2-axis | ratio |")
    p("| --- | --- | --- | --- |")
    p(
        f"| mean daily coverage of a random ISS-belt cluster | "
        f"{100.0 * e['mean_cov1']:.2f}% | {100.0 * e['mean_cov2']:.2f}% | "
        f"{e['mean_cov2'] / max(1e-12, e['mean_cov1']):.2f}x |"
    )
    p(
        f"| yield proxy E[T x coverage] (s x frac) | {e['e_yield1']:.2f} | "
        f"{e['e_yield2']:.2f} | {e['e_yield2'] / max(1e-12, e['e_yield1']):.2f}x |"
    )
    p("")
    p("So even where Earth-rotation azimuth walk is ~0, dropping the azimuth")
    p("axis throws away a factor of ~6 in how much industrial area a given")
    p("day can put in the frame. That is the lateral-motion cost of one axis,")
    p("independent of the along-track 124 s window.")
    p("")
    p("## Decision numbers")
    p("")
    p("1. **Along-track, given a pass over the origin:** stack-weighted expected")
    p(f"   loss is **{e['e_lost']:.0f} s of {e['e_t2']:.0f} s** ({e['e_lost_pct']:.0f}%).")
    p("   Almost all of that is the 20-40 N band (China, US, Med, India,")
    p("   Japan/Korea), where Earth rotation walks the covering disk out at")
    p("   the 30 deg stop. Immediate leftover hunt at the imaging rewind cap")
    p(
        f"   recovers that to **{e['e_lost_reacq']:.0f} s lost** "
        f"({e['e_lost_reacq_pct']:.0f}%) in this ocean-averaged model."
    )
    p("   Only ~10% of stacks sit at |lat| >= 45 deg.")
    p("2. **Lateral, whether we acquire at all:** 2-axis covers ~6x more")
    p("   industrial area per day. One axis only works plants that sit under")
    p("   the 24 km ground-track ribbon. Combined with shorter tracking time,")
    p(
        f"   the yield proxy E[T x coverage] is "
        f"**{e['e_yield2'] / max(1e-12, e['e_yield1']):.1f}x** higher with two axes."
    )
    p("3. A polar-slice one-axis payload that is only tasked at |lat| >= 45 deg")
    p("   keeps the 124 s window, but it is looking at ~10% of the world's")
    p("   stack-bearing industry. If the mission must work the 20-40 N")
    p("   belt, 2-axis is required for the elevation window, not just swath.")
    p("")
    p("## Figures")
    p("")
    p("- `outputs/industrial_lat_hist.png`")
    p("- `outputs/industrial_lat_folded.png`")
    p("- `outputs/industrial_time_vs_lat.png`")
    p("- `outputs/industrial_cluster_radius.png`")
    p("- `outputs/industrial_r_vs_lat.png`")
    p("- `outputs/industrial_cluster_map.png`")
    p("- `outputs/industrial_lat_hist.csv`, `outputs/industrial_clusters.csv`")
    p("- `outputs/r_vs_lat.csv` -- D(|lat|), R(|lat|), singleton fraction")
    p("- `outputs/expected_tracking.csv` -- includes T1_eff / lost-reacq columns")
    p("- `outputs/time_lost_grid.csv` -- the `time_lost(|lat|, R)` table")
    p("")
    p("```text")
    p(f"{_RUN_CMD} industry")
    p("```")
    p("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_study_readme(
    optics: Optics,
    times: WindowTimes,
    omega_img: float,
    exp: dict[str, dict[str, float]],
    folded: list[LatBandRow],
    path: Path,
) -> None:
    """Write the study-folder README.md from geometry + industry summaries.

    Args:
        optics: Usable sensor FOV.
        times: Design-pass elevation window.
        omega_img: Imaging rewind cap in degrees per second.
        exp: Expected-time summaries keyed by weight kind.
        folded: Coarse |lat| band rows.
        path: Output markdown path.

    Returns:
        None.
    """
    h_km = times.local_alt_km
    r_fit = h_km * math.tan(math.radians(optics.half_az_deg))
    r_box = h_km * math.tan(math.radians(GIMBAL_BOX.az_box_deg))
    e = exp["stacks"]
    omega_rel = MAX_SMEAR_BAND_PX * optics.ifov_band_deg / EXPOSURE_S
    lines: list[str] = []
    a = lines.append
    a("# Single-axis vs dual-axis gimbal")
    a("")
    a("TEMPORARY ANALYSIS. Not flight software. Design study for dropping")
    a("the azimuth gimbal axis. Shared geometry lives in `analysis.lib`;")
    a("this folder holds generated reports, CSV, and local inventory downloads.")
    a("")
    a("```text")
    a(f"{_RUN_CMD} geometry")
    a(f"{_RUN_CMD} industry")
    a(f"{_RUN_CMD} all")
    a("```")
    a("")
    a("Geometry: [`RESULTS.md`](RESULTS.md). Worldwide stack inventory:")
    a("[`INDUSTRIAL.md`](INDUSTRIAL.md) and [`outputs/`](outputs/).")
    a("")
    lines.extend(_setup_lines(include_heading=True))
    a("")
    a(f"## Headline (design pass, lat {DESIGN_LAT_DEG:.2f} deg, h = {h_km:.0f} km)")
    a("")
    a("| Quantity | Value |")
    a("| --- | --- |")
    a(f"| Camera / lens | {CAMERA_NAME} + 150 mm, {SENSOR_NAME} |")
    a(f"| Usable FOV | {optics.fov_az_deg:.2f} deg x {optics.fov_el_deg:.2f} deg |")
    a(f"| Band plane | {BAND_LATERAL_PX} x {BAND_ALONG_PX} (IFOV x2) |")
    a(f"| Along-track track time | **{times.along_track_s:.0f} s** |")
    a(f"| Staring, no gimbal | {times.stare_s:.1f} s |")
    a(f"| Peak elevation rate | {times.peak_el_rate_deg_s:.2f} deg/s |")
    a(f"| Imaging rewind (current gate) | **{omega_img:.2f} deg/s** |")
    a(f"| Scene-relative smear limit | {omega_rel:.2f} deg/s |")
    a(f"| Hardware slew cap | {MAX_HW_SLEW_DEG_S:.0f} deg/s |")
    a(f"| Nadir lateral half-swath | +/-{r_fit:.1f} km |")
    a(f"| 2-axis +/-{GIMBAL_BOX.az_box_deg:.0f} deg half-swath | +/-{r_box:.0f} km |")
    a(f"| Earth-rotation az walk of the origin | {times.az_max_deg:.3f} deg |")
    a("")
    a("At the design pass the origin stays in the chip. There is no locked")
    a("design R. Mid-latitude loss is set by Earth rotation plus R(|lat|)")
    a("from the inventory, not by a 5 km placeholder.")
    a("")
    a("## Worldwide stack distribution (Climate TRACE 2025)")
    a("")
    a("| Quantity | Value |")
    a("| --- | --- |")
    a(f"| ISS-belt stack fraction | {100.0 * e['frac_iss_of_world']:.1f}% |")
    a(f"| Stack-weighted mean R | {e['mean_r_km']:.2f} km (inventory, not design) |")
    a(f"| Stacks at |lat| >= 45 deg | **{100.0 * e['frac_lat_ge_45']:.0f}%** |")
    a(
        f"| E[T 1-axis] vs 2-axis, stack-weighted | "
        f"**{e['e_t1']:.0f} s vs {e['e_t2']:.0f} s ({e['e_lost_pct']:.0f}% lost)** |"
    )
    a(
        f"| After leftover hunt at omega_img | "
        f"**{e['e_t1_reacq']:.0f} s vs {e['e_t2']:.0f} s "
        f"({e['e_lost_reacq_pct']:.0f}% lost)** |"
    )
    a(
        f"| Daily in-swath yield (T x coverage) | "
        f"**~{e['e_yield2'] / max(1e-12, e['e_yield1']):.0f}x** in favour of 2-axis |"
    )
    a("")
    a("| |lat| (deg) | D (km) | R (km) | singleton % | d_char n>=2 (km) |")
    a("| --- | --- | --- | --- | --- |")
    for row in folded:
        a(
            f"| {row.lat_lo:.0f}-{row.lat_hi:.1f} | {row.mean_d_km:.2f} | "
            f"{row.mean_r_km:.2f} | {100.0 * row.frac_singleton_stacks:.0f}% | "
            f"{row.d_char_n2_km:.2f} |"
        )
    a("")
    a("Most stacks sit at 20-40 N. A polar-slice one-axis view keeps the")
    a(f"{times.along_track_s:.0f} s window but sees only ~10% of the industry.")
    a("Working the mid-latitude belt needs the azimuth axis for Earth-rotation")
    a("walk, not just for swath. Leftover hunt during the slew recovers some")
    a("one-axis time only if another cluster sits on the FOV ribbon.")
    a("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
