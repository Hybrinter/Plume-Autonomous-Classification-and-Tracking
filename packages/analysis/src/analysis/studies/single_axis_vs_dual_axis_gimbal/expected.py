"""Stack-weighted expected tracking time over the ISS belt.

Folds ``TimeLostFn`` over Climate TRACE clusters. Primary weight is stack
count. Covering radius is per-cluster R = D + L and, for lat curves,
stack-weighted R(|lat|). After a target leaves the frame the gimbal rewinds
toward the science limb stop, searching along that path, then (two-axis)
rasters azimuth. Lost time is that two-phase reacquire from signed-latitude
stack density. Cycle time is dwell plus reacquire. Primary yield is
stack-weighted inferred-usable plume-seconds per day (dwell gated by
``T_MIN_USABLE_S``, times daily coverage).

Contains:
  - iss_dwell_weight / daily_coverage_frac / expected_from_clusters.
  - run_industry: ingest, cluster, interpolate, report.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from analysis.lib.hunt import HuntModel
from analysis.lib.optics import Optics, build_optics
from analysis.lib.orbit import Orbit, build_orbit
from analysis.lib.plot_style import apply as apply_plot_style
from analysis.lib.tracking import SampleSpan, TimeLostFn
from analysis.studies.single_axis_vs_dual_axis_gimbal.assumptions import (
    CACHE_DIR,
    GIMBAL_BOX,
    GRID_R_KM,
    INDUSTRY_PASS_DT_S,
    LAT_BIN_DEG,
    OPTICS_SPEC,
    OUT_DIR,
    PLUME_L_PERCENTILES,
    STUDY_DIR,
    T_MIN_USABLE_S,
    TLE,
    grid_lats_deg,
    omega_img_rewind_deg_s,
    omega_rel_max_deg_s,
)
from analysis.studies.single_axis_vs_dual_axis_gimbal.geometry import origin_window
from analysis.studies.single_axis_vs_dual_axis_gimbal.inventory import (
    Cluster,
    build_clusters,
    fetch_gem,
    load_climate_trace,
    load_clusters_csv,
    load_gppd,
)
from analysis.studies.single_axis_vs_dual_axis_gimbal.profile import (
    RadiusProfile,
    build_radius_profile,
    cycle_s,
    folded_band_rows,
    hunt_at_lat,
)

Weight = Literal["area", "stacks", "emissions"]


def iss_dwell_weight(lat_deg: float, i_deg: float | None = None) -> float:
    """Return relative time spent near |lat| on a circular inclined orbit.

    dt/dphi is proportional to cos phi / sqrt(sin^2 i - sin^2 phi).
    Diverges at +/- i and is clipped.

    Args:
        lat_deg: Geocentric latitude in degrees.
        i_deg: Inclination in degrees. Default is the study TLE.

    Returns:
        Relative dwell weight (not normalised).
    """
    inc = TLE.inclination_deg if i_deg is None else i_deg
    phi = math.radians(abs(lat_deg))
    inc_rad = math.radians(inc)
    s2 = math.sin(inc_rad) ** 2 - math.sin(phi) ** 2
    if s2 <= 1e-6:
        s2 = 1e-6
    return math.cos(phi) / math.sqrt(s2)


def daily_coverage_frac(
    lat_deg: float, half_swath_km: float, n_rev_day: float, re_km: float
) -> float:
    """Return the fraction of a parallel covered per day by asc+desc passes.

    Args:
        lat_deg: Latitude in degrees.
        half_swath_km: Ground half-swath in kilometres.
        n_rev_day: Mean motion in revolutions per day.
        re_km: Earth radius in kilometres.

    Returns:
        Coverage fraction in [0, 1].
    """
    cphi = max(0.08, math.cos(math.radians(lat_deg)))
    circ = 2.0 * math.pi * re_km * cphi
    covered = n_rev_day * 2.0 * (2.0 * half_swath_km)
    return min(1.0, covered / circ)


def histogram(lats: np.ndarray, weights: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return a weighted latitude histogram.

    Args:
        lats: Latitudes in degrees. # np.ndarray[float64, (N,)]
        weights: Bin weights. # np.ndarray[float64, (N,)]
        edges: Bin edges in degrees. # np.ndarray[float64, (B+1,)]

    Returns:
        Counts per bin. # np.ndarray[float64, (B,)]
    """
    hist, _ = np.histogram(lats, bins=edges, weights=weights)
    return hist


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted mean, or 0 if the weight sum is 0.

    Args:
        values: Samples. # np.ndarray[float64, (N,)]
        weights: Non-negative weights. # np.ndarray[float64, (N,)]

    Returns:
        Weighted mean.
    """
    w_sum = float(np.sum(weights))
    if w_sum <= 0:
        return 0.0
    return float(np.sum(values * weights) / w_sum)


def weighted_mean_finite(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted mean of finite samples, or inf if none are finite.

    Args:
        values: Samples, possibly inf/nan. # np.ndarray[float64, (N,)]
        weights: Non-negative weights. # np.ndarray[float64, (N,)]

    Returns:
        Weighted mean of finite entries, or inf.
    """
    mask = np.isfinite(values)
    if not np.any(mask):
        return math.inf
    return weighted_mean(values[mask], weights[mask])


def _weight_of(cluster: Cluster, weight: Weight) -> float:
    """Return the selected weight of one cluster.

    Args:
        cluster: Plant cluster.
        weight: Weight kind.

    Returns:
        Weight value.
    """
    if weight == "area":
        return cluster.area_km2
    if weight == "stacks":
        return float(cluster.n)
    return cluster.emissions_t


def expected_from_clusters(
    clusters: list[Cluster],
    fn: TimeLostFn,
    optics: Optics,
    orbit: Orbit,
    weight: Weight,
    profile: RadiusProfile,
    hunt: HuntModel,
) -> dict[str, float]:
    """Return expected dwell, reacquire, and cycle times over ISS-belt clusters.

    Single-target dwell is T1 / T2 at each cluster's own covering radius.
    After that target leaves, the gimbal rewinds toward the limb stop and
    searches; two-axis then rasters azimuth. Lost time is mean reacquire
    from stack density at that signed latitude. Cycle time is dwell plus
    reacquire. Off-track origins appear in the swath columns, not dwell.

    Args:
        clusters: All world clusters (ISS-belt filter applied here).
        fn: Tracking-time interpolator.
        optics: Usable sensor FOV.
        orbit: Circular ISS orbit.
        weight: Weight kind.
        profile: R(|lat|) and signed-latitude stack-density interpolator.
        hunt: Rewind-then-scan hunt model.

    Returns:
        Summary dict of means and fractions. Empty if no ISS-belt clusters.
    """
    iss = [c for c in clusters if abs(c.lat) <= orbit.inclination_deg]
    if not iss:
        return {}
    lats = np.array([c.lat for c in iss])
    rs = np.array([c.r_cover_km for c in iss])
    w = np.array([_weight_of(c, weight) for c in iss])
    t1 = np.zeros(len(iss))
    t2 = np.zeros(len(iss))
    reacq1 = np.zeros(len(iss))
    reacq2 = np.zeros(len(iss))
    cyc1 = np.zeros(len(iss))
    cyc2 = np.zeros(len(iss))
    for i, cluster in enumerate(iss):
        a_t, b_t, _lost = fn.eval(cluster.lat, cluster.r_cover_km)
        t1[i] = a_t
        t2[i] = b_t
        t_window = fn.eval(cluster.lat, float(fn.rs[0]))[1]
        hunted = hunt_at_lat(
            cluster.lat,
            profile.dens_km2(cluster.lat),
            hunt,
            t_dwell_1_s=a_t,
            t_dwell_2_s=b_t,
            t_window_s=t_window,
        )
        reacq1[i] = hunted.t_reacq_1_s
        reacq2[i] = hunted.t_reacq_2_s
        cyc1[i] = cycle_s(a_t, hunted.t_reacq_1_s)
        cyc2[i] = cycle_s(b_t, hunted.t_reacq_2_s)
    lost_same = t2 - t1
    duty1 = np.divide(t1, cyc1, out=np.zeros_like(t1), where=np.isfinite(cyc1) & (cyc1 > 0))
    duty2 = np.divide(t2, cyc2, out=np.zeros_like(t2), where=np.isfinite(cyc2) & (cyc2 > 0))
    re_km = orbit.earth_radius_km(0.0)
    n_rev = TLE.mean_motion_rev_per_day
    h_local = np.array([orbit.local_altitude_km(abs(c.lat)) for c in iss])
    half1 = h_local * math.tan(math.radians(optics.half_az_deg))
    half2 = h_local * math.tan(math.radians(GIMBAL_BOX.az_box_deg))
    cov1 = np.array(
        [
            daily_coverage_frac(abs(c.lat), float(h1), n_rev, re_km)
            for c, h1 in zip(iss, half1, strict=True)
        ]
    )
    cov2 = np.array(
        [
            daily_coverage_frac(abs(c.lat), float(h2), n_rev, re_km)
            for c, h2 in zip(iss, half2, strict=True)
        ]
    )
    dwell = np.array([iss_dwell_weight(c.lat) for c in iss])
    high = np.abs(lats) >= 45.0
    world_w = sum(_weight_of(c, weight) for c in clusters)
    e_t1 = weighted_mean(t1, w)
    e_t2 = weighted_mean(t2, w)
    e_reacq1 = weighted_mean_finite(reacq1, w)
    e_reacq2 = weighted_mean_finite(reacq2, w)
    e_cyc1 = weighted_mean_finite(cyc1, w)
    e_cyc2 = weighted_mean_finite(cyc2, w)
    usable1 = np.where(t1 >= T_MIN_USABLE_S, t1, 0.0)
    usable2 = np.where(t2 >= T_MIN_USABLE_S, t2, 0.0)
    visit1 = (t1 >= T_MIN_USABLE_S).astype(float)
    visit2 = (t2 >= T_MIN_USABLE_S).astype(float)
    return {
        "n_clusters": float(len(iss)),
        "weight_sum": float(np.sum(w)),
        "frac_iss_of_world": float(np.sum(w) / max(1e-12, world_w)),
        "frac_lat_ge_45": float(np.sum(w[high]) / max(1e-12, np.sum(w))),
        "mean_r_km": weighted_mean(rs, w),
        "mean_n": weighted_mean(np.array([float(c.n) for c in iss]), w),
        "e_t1": e_t1,
        "e_t2": e_t2,
        "e_lost_same": weighted_mean(lost_same, w),
        "e_lost_same_pct": 100.0 * weighted_mean(lost_same, w) / max(1e-12, e_t2),
        "e_reacq1": e_reacq1,
        "e_reacq2": e_reacq2,
        "e_cycle1": e_cyc1,
        "e_cycle2": e_cyc2,
        "e_duty1": weighted_mean(duty1, w),
        "e_duty2": weighted_mean(duty2, w),
        "e_t1_dwell": weighted_mean(t1, w * dwell),
        "e_t2_dwell": weighted_mean(t2, w * dwell),
        "e_lost_dwell": weighted_mean(lost_same, w * dwell),
        "e_yield1": weighted_mean(t1 * cov1, w),
        "e_yield2": weighted_mean(t2 * cov2, w),
        "e_usable_yield1": weighted_mean(usable1 * cov1, w),
        "e_usable_yield2": weighted_mean(usable2 * cov2, w),
        "e_visit1": weighted_mean(visit1 * cov1, w),
        "e_visit2": weighted_mean(visit2 * cov2, w),
        "mean_cov1": weighted_mean(cov1, w),
        "mean_cov2": weighted_mean(cov2, w),
        "omega_img_deg_s": hunt.omega_img_deg_s,
        "t_min_usable_s": T_MIN_USABLE_S,
    }


def run_industry() -> None:
    """Load inventories, cluster, interpolate T(lat, R), write INDUSTRIAL.md.

    Returns:
        None. Writes figures and CSV under OUT_DIR.

    Raises:
        FileNotFoundError: If Climate TRACE CSVs, cache, and cluster CSV are missing.
    """
    from analysis.studies.single_axis_vs_dual_axis_gimbal.figures import (
        plot_expected_vs_lat,
        plot_folded,
        plot_lat_hist,
        plot_map,
        plot_r_hist,
        plot_r_vs_lat,
        plot_reacquire_vs_lat,
        plot_reacquire_vs_lat_plume_length,
    )
    from analysis.studies.single_axis_vs_dual_axis_gimbal.report import (
        write_industry_report,
        write_study_readme,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    apply_plot_style()
    optics = build_optics(OPTICS_SPEC)
    orbit = build_orbit(TLE, use_perigee=False)
    times, _ = origin_window(orbit, optics, GIMBAL_BOX, TLE.inclination_deg, 0.0)
    omega_rel = omega_rel_max_deg_s(optics.ifov_band_deg)
    omega_img = omega_img_rewind_deg_s(optics.ifov_band_deg, times.peak_el_rate_deg_s)
    hunt = HuntModel(
        optics=optics,
        orbit=orbit,
        box=GIMBAL_BOX,
        omega_img_deg_s=omega_img,
        omega_rel_deg_s=omega_rel,
    )

    print("loading Climate TRACE ...")
    try:
        ct = load_climate_trace()
        print(f"CT sources {len(ct)}")
        clusters = build_clusters(ct)
        ct_n = len(ct)
    except FileNotFoundError as exc:
        csv_path = OUT_DIR / "industrial_clusters.csv"
        print(f"Climate TRACE missing ({exc}); using {csv_path}")
        clusters = load_clusters_csv(csv_path)
        ct_n = sum(c.n for c in clusters)
    print("loading GPPD / GEM (cross-check) ...")
    gppd = load_gppd()
    try:
        gem = fetch_gem()
    except Exception as exc:
        print("GEM fetch failed:", exc)
        gem = []
    print(f"GPPD {len(gppd)}  GEM {len(gem)}")
    print(f"clusters {len(clusters)}")
    profile = build_radius_profile(clusters)
    folded = folded_band_rows(clusters)

    lats_grid = np.array(grid_lats_deg(), dtype=float)
    rs_grid = np.array(GRID_R_KM, dtype=float)
    cache_path = CACHE_DIR / "time_lost_grid.npz"
    fn = TimeLostFn(
        orbit,
        optics,
        GIMBAL_BOX,
        lats_grid,
        rs_grid,
        cache_path,
        span=SampleSpan(dt_s=INDUSTRY_PASS_DT_S),
        verbose=True,
    )

    edges = np.arange(-90.0, 90.0 + LAT_BIN_DEG, LAT_BIN_DEG)
    lats = np.array([c.lat for c in clusters])
    h_area = histogram(lats, np.array([c.area_km2 for c in clusters]), edges)
    h_stacks = histogram(lats, np.array([float(c.n) for c in clusters]), edges)
    h_em = histogram(lats, np.array([c.emissions_t for c in clusters]), edges)

    h_gppd = np.zeros(edges.size - 1)
    if gppd:
        g_lats = np.array([f.lat for f in gppd])
        h_gppd, _ = np.histogram(g_lats, bins=edges)

    t1_bin = np.zeros(edges.size - 1)
    t2_bin = np.zeros(edges.size - 1)
    r_bin = np.zeros(edges.size - 1)
    reacq1_bin = np.zeros(edges.size - 1)
    reacq2_bin = np.zeros(edges.size - 1)
    cyc1_bin = np.zeros(edges.size - 1)
    cyc2_bin = np.zeros(edges.size - 1)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mid = 0.5 * (lo + hi)
        if abs(mid) > orbit.inclination_deg:
            continue
        radius = profile.r_km(mid)
        r_bin[i] = radius
        t1_bin[i], t2_bin[i], _ = fn.eval(mid, radius)
        hunted = hunt_at_lat(
            mid,
            profile.dens_km2(mid),
            hunt,
            t_dwell_1_s=t1_bin[i],
            t_dwell_2_s=t2_bin[i],
            t_window_s=t2_bin[i],
        )
        reacq1_bin[i] = hunted.t_reacq_1_s
        reacq2_bin[i] = hunted.t_reacq_2_s
        cyc1_bin[i] = cycle_s(t1_bin[i], hunted.t_reacq_1_s)
        cyc2_bin[i] = cycle_s(t2_bin[i], hunted.t_reacq_2_s)

    t1_by_label: dict[str, np.ndarray] = {}
    reacq1_by_label: dict[str, np.ndarray] = {}
    n_bins = edges.size - 1
    for pct_name, l_km in PLUME_L_PERCENTILES:
        t1_l = np.zeros(n_bins)
        reacq1_l = np.zeros(n_bins)
        curve = f"{pct_name} L={l_km:g} km"
        for i, mid in enumerate(0.5 * (edges[:-1] + edges[1:])):
            if abs(mid) > orbit.inclination_deg:
                continue
            radius = profile.r_km(mid, plume_r_km=l_km)
            t1_l[i], _t2_l, _lost = fn.eval(mid, radius)
            hunted_l = hunt_at_lat(
                mid,
                profile.dens_km2(mid),
                hunt,
                t_dwell_1_s=t1_l[i],
                t_dwell_2_s=t2_bin[i],
                t_window_s=t2_bin[i],
            )
            reacq1_l[i] = hunted_l.t_reacq_1_s
        t1_by_label[curve] = t1_l
        reacq1_by_label[curve] = reacq1_l

    weights: tuple[Weight, ...] = ("area", "stacks", "emissions")
    exp: dict[str, dict[str, float]] = {}
    for key in weights:
        exp[key] = expected_from_clusters(clusters, fn, optics, orbit, key, profile, hunt)
    for weight_name, summary in exp.items():
        print(
            f"{weight_name:10s}  dwell T1={summary['e_t1']:.1f}s T2={summary['e_t2']:.1f}s  "
            f"reacq 1={summary['e_reacq1']:.1f}s 2={summary['e_reacq2']:.1f}s  "
            f"cycle 1={summary['e_cycle1']:.1f}s 2={summary['e_cycle2']:.1f}s  "
            f"|lat|>=45 {100 * summary['frac_lat_ge_45']:.1f}%"
        )

    plot_lat_hist(edges, h_area, h_stacks, h_em, h_gppd, OUT_DIR / "industrial_lat_hist.png")
    plot_expected_vs_lat(edges, t1_bin, t2_bin, h_stacks, OUT_DIR / "industrial_time_vs_lat.png")
    plot_reacquire_vs_lat(
        edges,
        t1_bin,
        t2_bin,
        reacq1_bin,
        reacq2_bin,
        h_stacks,
        OUT_DIR / "industrial_reacquire_vs_lat.png",
    )
    plot_reacquire_vs_lat_plume_length(
        edges,
        t2_bin,
        reacq2_bin,
        t1_by_label,
        reacq1_by_label,
        h_stacks,
        OUT_DIR / "industrial_reacquire_vs_lat_plume_length.png",
    )
    plot_folded(clusters, fn, profile, OUT_DIR / "industrial_lat_folded.png")
    plot_r_hist(clusters, OUT_DIR / "industrial_cluster_radius.png")
    plot_r_vs_lat(profile, OUT_DIR / "industrial_r_vs_lat.png")
    plot_map(clusters, OUT_DIR / "industrial_cluster_map.png")

    with (OUT_DIR / "industrial_lat_hist.csv").open("w", encoding="utf-8") as fh:
        fh.write(
            "lat_lo,lat_hi,lat_mid,area_km2,n_stacks,emissions_t,r_km,"
            "one_axis_s,two_axis_s,stack_dens_per_km2,reacq1_s,reacq2_s,"
            "cycle1_s,cycle2_s\n"
        )
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            mid = 0.5 * (lo + hi)
            dens = profile.dens_km2(mid) if abs(mid) <= orbit.inclination_deg else 0.0
            fh.write(
                f"{lo:.1f},{hi:.1f},{mid:.1f},{h_area[i]:.4f},{h_stacks[i]:.4f},"
                f"{h_em[i]:.4f},{r_bin[i]:.3f},{t1_bin[i]:.3f},{t2_bin[i]:.3f},"
                f"{dens:.8g},{reacq1_bin[i]:.3f},{reacq2_bin[i]:.3f},"
                f"{cyc1_bin[i]:.3f},{cyc2_bin[i]:.3f}\n"
            )
    with (OUT_DIR / "industrial_reacquire_vs_lat_plume_length.csv").open(
        "w", encoding="utf-8"
    ) as fh:
        header = "lat_lo,lat_hi,lat_mid,two_axis_dwell_s,two_axis_reacq_s"
        for curve in t1_by_label:
            slug = curve.replace(" ", "_").replace("=", "")
            header += f",{slug}_dwell_s,{slug}_reacq_s"
        fh.write(header + "\n")
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            mid = 0.5 * (lo + hi)
            line = f"{lo:.1f},{hi:.1f},{mid:.1f},{t2_bin[i]:.3f},{reacq2_bin[i]:.3f}"
            for curve in t1_by_label:
                line += f",{t1_by_label[curve][i]:.3f},{reacq1_by_label[curve][i]:.3f}"
            fh.write(line + "\n")
    with (OUT_DIR / "r_vs_lat.csv").open("w", encoding="utf-8") as fh:
        fh.write(
            "lat_lo,lat_hi,lat_mid,n_clusters,n_stacks,mean_d_km,mean_r_km,"
            "mean_n,frac_singleton_stacks,d_char_n2_km,dens_per_km2\n"
        )
        for row in profile.rows:
            fh.write(
                f"{row.lat_lo:.3f},{row.lat_hi:.3f},{row.lat_mid:.3f},"
                f"{row.n_clusters},{row.n_stacks},{row.mean_d_km:.4f},"
                f"{row.mean_r_km:.4f},{row.mean_n:.4f},{row.frac_singleton_stacks:.4f},"
                f"{row.d_char_n2_km:.4f},{row.dens_per_km2:.8g}\n"
            )
    with (OUT_DIR / "industrial_clusters.csv").open("w", encoding="utf-8") as fh:
        fh.write("lat,lon,n,r_plant_km,r_cover_km,area_km2,emissions_t,capacity,subsector\n")
        for cluster in clusters:
            fh.write(
                f"{cluster.lat:.5f},{cluster.lon:.5f},{cluster.n},"
                f"{cluster.r_plant_km:.3f},{cluster.r_cover_km:.3f},"
                f"{cluster.area_km2:.4f},{cluster.emissions_t:.4f},"
                f"{cluster.capacity:.4f},{cluster.subsector}\n"
            )
    with (OUT_DIR / "time_lost_grid.csv").open("w", encoding="utf-8") as fh:
        fh.write("lat_deg,radius_km,one_axis_s,two_axis_s,lost_s\n")
        for i, lat in enumerate(fn.lats):
            for j, radius in enumerate(fn.rs):
                t1 = float(fn.t1_grid[i, j])
                t2 = float(fn.t2_grid[i, j])
                fh.write(f"{lat:.4f},{radius:.3f},{t1:.3f},{t2:.3f},{t2 - t1:.3f}\n")
    with (OUT_DIR / "expected_tracking.csv").open("w", encoding="utf-8") as fh:
        keys = list(next(iter(exp.values())).keys())
        fh.write("weight," + ",".join(keys) + "\n")
        for name, summary in exp.items():
            fh.write(name + "," + ",".join(f"{summary[k]:.6g}" for k in keys) + "\n")

    write_industry_report(
        clusters,
        len(gem),
        len(gppd),
        ct_n,
        fn,
        exp,
        profile,
        folded,
        hunt,
        STUDY_DIR / "INDUSTRIAL.md",
    )
    write_study_readme(
        optics,
        times,
        omega_img,
        exp,
        folded,
        STUDY_DIR / "README.md",
    )
    print(f"wrote {OUT_DIR} and INDUSTRIAL.md")
