"""Stack-weighted expected tracking time over the ISS belt.

Folds ``TimeLostFn`` over Climate TRACE clusters. Primary weight is stack
count. Dwell (time spent near +/- i) and daily swath coverage are
sensitivities. ``run_industry`` writes CSV, figures, and INDUSTRIAL.md.

Contains:
  - iss_dwell_weight / daily_coverage_frac / expected_from_clusters.
  - run_industry: ingest, cluster, interpolate, report.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from analysis.lib.optics import Optics, build_optics
from analysis.lib.orbit import Orbit, build_orbit
from analysis.lib.plot_style import apply as apply_plot_style
from analysis.lib.tracking import SampleSpan, TimeLostFn
from analysis.studies.single_axis_vs_dual_axis_gimbal.assumptions import (
    CACHE_DIR,
    CLUSTER_R_KM,
    GIMBAL_BOX,
    GRID_R_KM,
    INDUSTRY_PASS_DT_S,
    LAT_BIN_DEG,
    OPTICS_SPEC,
    OUT_DIR,
    STUDY_DIR,
    TLE,
    grid_lats_deg,
)
from analysis.studies.single_axis_vs_dual_axis_gimbal.inventory import (
    Cluster,
    build_clusters,
    fetch_gem,
    load_climate_trace,
    load_gppd,
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
) -> dict[str, float]:
    """Return expected tracking times over ISS-belt clusters.

    A pass is assumed over each cluster CoG. Off-track origins are not in
    the tracking-time expectation; they appear in the swath columns.

    Args:
        clusters: All world clusters (ISS-belt filter applied here).
        fn: Tracking-time interpolator.
        optics: Usable sensor FOV.
        orbit: Circular ISS orbit.
        weight: Weight kind.

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
    for i, cluster in enumerate(iss):
        a_t, b_t, _lost = fn.eval(cluster.lat, cluster.r_cover_km)
        t1[i] = a_t
        t2[i] = b_t
    lost = t2 - t1
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
    return {
        "n_clusters": float(len(iss)),
        "weight_sum": float(np.sum(w)),
        "frac_iss_of_world": float(np.sum(w) / max(1e-12, world_w)),
        "frac_lat_ge_45": float(np.sum(w[high]) / max(1e-12, np.sum(w))),
        "mean_r_km": weighted_mean(rs, w),
        "mean_n": weighted_mean(np.array([float(c.n) for c in iss]), w),
        "e_t1": weighted_mean(t1, w),
        "e_t2": weighted_mean(t2, w),
        "e_lost": weighted_mean(lost, w),
        "e_lost_pct": 100.0 * weighted_mean(lost, w) / max(1e-12, weighted_mean(t2, w)),
        "e_t1_dwell": weighted_mean(t1, w * dwell),
        "e_t2_dwell": weighted_mean(t2, w * dwell),
        "e_lost_dwell": weighted_mean(lost, w * dwell),
        "e_yield1": weighted_mean(t1 * cov1, w),
        "e_yield2": weighted_mean(t2 * cov2, w),
        "mean_cov1": weighted_mean(cov1, w),
        "mean_cov2": weighted_mean(cov2, w),
    }


def run_industry() -> None:
    """Load inventories, cluster, interpolate T(lat, R), write INDUSTRIAL.md.

    Returns:
        None. Writes figures and CSV under OUT_DIR.

    Raises:
        FileNotFoundError: If Climate TRACE CSVs and cache are both missing.
    """
    from analysis.studies.single_axis_vs_dual_axis_gimbal.figures import (
        plot_expected_vs_lat,
        plot_folded,
        plot_lat_hist,
        plot_map,
        plot_r_hist,
    )
    from analysis.studies.single_axis_vs_dual_axis_gimbal.report import write_industry_report

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    apply_plot_style()
    optics = build_optics(OPTICS_SPEC)
    orbit = build_orbit(TLE, use_perigee=False)

    print("loading Climate TRACE ...")
    ct = load_climate_trace()
    print(f"CT sources {len(ct)}")
    print("loading GPPD / GEM (cross-check) ...")
    gppd = load_gppd()
    try:
        gem = fetch_gem()
    except Exception as exc:
        print("GEM fetch failed:", exc)
        gem = []
    print(f"GPPD {len(gppd)}  GEM {len(gem)}")

    clusters = build_clusters(ct)
    print(f"clusters {len(clusters)}")

    fn = TimeLostFn(
        orbit,
        optics,
        GIMBAL_BOX,
        np.array(grid_lats_deg(), dtype=float),
        np.array(GRID_R_KM, dtype=float),
        CACHE_DIR / "time_lost_grid.npz",
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
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mid = 0.5 * (lo + hi)
        if abs(mid) > orbit.inclination_deg:
            continue
        t1_bin[i], t2_bin[i], _ = fn.eval(mid, CLUSTER_R_KM)

    weights: tuple[Weight, ...] = ("area", "stacks", "emissions")
    exp: dict[str, dict[str, float]] = {}
    for key in weights:
        exp[key] = expected_from_clusters(clusters, fn, optics, orbit, key)
    for weight_name, summary in exp.items():
        print(
            f"{weight_name:10s}  E[T1]={summary['e_t1']:.1f}s  E[T2]={summary['e_t2']:.1f}s  "
            f"lost={summary['e_lost']:.1f}s ({summary['e_lost_pct']:.1f}%)  "
            f"|lat|>=45 {100 * summary['frac_lat_ge_45']:.1f}%"
        )

    plot_lat_hist(edges, h_area, h_stacks, h_em, h_gppd, OUT_DIR / "industrial_lat_hist.png")
    plot_expected_vs_lat(edges, t1_bin, t2_bin, h_stacks, OUT_DIR / "industrial_time_vs_lat.png")
    plot_folded(clusters, fn, OUT_DIR / "industrial_lat_folded.png")
    plot_r_hist(clusters, OUT_DIR / "industrial_cluster_radius.png")
    plot_map(clusters, OUT_DIR / "industrial_cluster_map.png")

    with (OUT_DIR / "industrial_lat_hist.csv").open("w", encoding="utf-8") as fh:
        fh.write(
            "lat_lo,lat_hi,lat_mid,area_km2,n_stacks,emissions_t,one_axis_s_R5,two_axis_s_R5\n"
        )
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            mid = 0.5 * (lo + hi)
            fh.write(
                f"{lo:.1f},{hi:.1f},{mid:.1f},{h_area[i]:.4f},{h_stacks[i]:.4f},"
                f"{h_em[i]:.4f},{t1_bin[i]:.3f},{t2_bin[i]:.3f}\n"
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
        len(ct),
        fn,
        exp,
        STUDY_DIR / "INDUSTRIAL.md",
    )
    print(f"wrote {OUT_DIR} and INDUSTRIAL.md")
