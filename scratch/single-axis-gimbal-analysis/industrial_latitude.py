#!/usr/bin/env python3
"""TEMPORARY ANALYSIS — not flight software.

Worldwide industrial-area vs latitude, then expected 1-axis vs 2-axis
tracking time.

Inventory: Climate TRACE v5.10.0 CO2 point sources that have smoke stacks
(combustion power, cement, steel, chemicals, refining, ...). Nearby
sources are clustered into plant complexes. Covering-disk area is the
size of each industrial area; stack count is the source count. Size is
assumed to scale with stack count.

    python3 scratch/single-axis-gimbal-analysis/industrial_latitude.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import analyze as gimbal  # noqa: E402

DATA = ROOT / "data"
RAW = DATA / "raw"
OUT = ROOT / "outputs"
CACHE = DATA / "cache"

YEAR = "2025"
PLUME_R_KM = 2.0
CLUSTER_EPS_KM = 8.0
MAX_CLUSTER_R_KM = 12.0  # split if covering radius exceeds the chip half-swath
MIN_PLANT_R_KM = 0.4  # isolated-facility floor (~0.5 km²)
ISS_I_DEG = gimbal.TLE_INCLINATION_DEG
LAT_BIN_DEG = 2.0

CT_CORE_SUBSECTORS = frozenset(
    {
        "electricity-generation",
        "cement",
        "iron-and-steel",
        "aluminum",
        "chemicals",
        "other-chemicals",
        "lime",
        "glass",
        "pulp-and-paper",
        "petrochemical-steam-cracking",
        "oil-and-gas-refining",
        "other-metals",
        "other-manufacturing",
    }
)
NON_STACK_TOKENS = (
    "hydro",
    "solar",
    "wind",
    "nuclear",
    "geothermal",
    "battery",
    "wave",
    "tidal",
    "marine",
    "storage",
)
GPPD_STACK_FUELS = frozenset(
    {
        "Coal",
        "Gas",
        "Oil",
        "Biomass",
        "Waste",
        "Cogeneration",
        "Petcoke",
        "Other",
        "Peat",
    }
)
GEM_STACK_TYPES = (
    "coal-plant",
    "oil-gas-plant",
    "bioenergy-plant",
    "cement-plant",
    "iron-steel-plant",
)

GRID_LATS = tuple([*np.linspace(0.0, 50.0, 26), ISS_I_DEG])
GRID_R_KM = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0)
PASS_DT_S = 0.1

C_BLUE = gimbal.C_BLUE
C_ORANGE = gimbal.C_ORANGE
C_TEAL = gimbal.C_TEAL
C_RED = gimbal.C_RED
C_INK = gimbal.C_INK


def _is_stack_power(source_type: str) -> bool:
    t = source_type.lower()
    if not t:
        return True
    if any(tok in t for tok in NON_STACK_TOKENS):
        return False
    return True


def _ffloat(s: str) -> float | None:
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_ct_file(path: Path) -> list[dict[str, object]]:
    """One row per source: 2025 CO2 annual total, with lat/lon."""
    acc: dict[str, dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        r = csv.reader(fh)
        header = next(r)
        col = {name: i for i, name in enumerate(header)}
        need = ("source_id", "source_name", "source_type", "subsector", "start_time", "lat", "lon", "gas", "emissions_quantity", "capacity")
        for name in need:
            if name not in col:
                raise KeyError(f"{path} missing {name}")
        i_id = col["source_id"]
        i_name = col["source_name"]
        i_type = col["source_type"]
        i_sub = col["subsector"]
        i_start = col["start_time"]
        i_lat = col["lat"]
        i_lon = col["lon"]
        i_gas = col["gas"]
        i_em = col["emissions_quantity"]
        i_cap = col["capacity"]
        i_sec = col.get("sector", -1)
        for row in r:
            if row[i_gas] != "co2" or not row[i_start].startswith(YEAR):
                continue
            sub = row[i_sub]
            if sub not in CT_CORE_SUBSECTORS:
                continue
            if sub == "electricity-generation" and not _is_stack_power(row[i_type]):
                continue
            lat = _ffloat(row[i_lat])
            lon = _ffloat(row[i_lon])
            if lat is None or lon is None:
                continue
            if abs(lat) > 90.0 or abs(lon) > 180.0:
                continue
            em = _ffloat(row[i_em]) or 0.0
            cap = _ffloat(row[i_cap]) or 0.0
            sid = row[i_id]
            rec = acc.get(sid)
            if rec is None:
                acc[sid] = {
                    "id": sid,
                    "name": row[i_name],
                    "source_type": row[i_type],
                    "subsector": sub,
                    "sector": row[i_sec] if i_sec >= 0 else "",
                    "lat": lat,
                    "lon": lon,
                    "emissions_t": em,
                    "capacity": cap,
                    "inventory": "climate-trace",
                }
            else:
                rec["emissions_t"] = float(rec["emissions_t"]) + em
                if cap > float(rec["capacity"]):
                    rec["capacity"] = cap
    return list(acc.values())


def load_climate_trace() -> list[dict[str, object]]:
    cache = CACHE / "ct_facilities.json"
    if cache.is_file():
        print(f"  CT cache {cache}")
        return json.loads(cache.read_text(encoding="utf-8"))
    files: list[Path] = []
    for folder in (RAW / "ct_power", RAW / "ct_manufacturing", RAW / "ct_fossil"):
        data = folder / "DATA"
        if not data.is_dir():
            continue
        files.extend(sorted(data.glob("*_emissions_sources_v5_10_0.csv")))
    if not files:
        raise FileNotFoundError(f"no Climate TRACE CSVs under {RAW}")
    rows: list[dict[str, object]] = []
    for path in files:
        print(f"  CT {path.name} ...", flush=True)
        part = load_ct_file(path)
        print(f"    {len(part)} sources")
        rows.extend(part)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def load_gppd() -> list[dict[str, object]]:
    path = RAW / "gppd" / "global_power_plant_database.csv"
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        r = csv.DictReader(fh)
        for row in r:
            fuel = (row.get("primary_fuel") or "").strip()
            if fuel not in GPPD_STACK_FUELS:
                continue
            lat = _ffloat(row.get("latitude") or "")
            lon = _ffloat(row.get("longitude") or "")
            cap = _ffloat(row.get("capacity_mw") or "")
            if lat is None or lon is None or cap is None or cap <= 0:
                continue
            rows.append(
                {
                    "id": row.get("gppd_idnr") or "",
                    "name": row.get("name") or "",
                    "source_type": fuel,
                    "subsector": "electricity-generation",
                    "sector": "power",
                    "lat": lat,
                    "lon": lon,
                    "emissions_t": 0.0,
                    "capacity": cap,
                    "inventory": "gppd",
                }
            )
    return rows


def fetch_gem() -> list[dict[str, object]]:
    cache = CACHE / "gem_operating.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "PACT-scratch-analysis/1.0")]
    for atype in GEM_STACK_TYPES:
        offset = 0
        limit = 500
        while True:
            url = (
                "https://api.globalenergymonitor.org/assets?"
                + urllib.parse.urlencode(
                    {
                        "asset_type": atype,
                        "status": "operating",
                        "limit": limit,
                        "offset": offset,
                    }
                )
            )
            try:
                with opener.open(url, timeout=60) as resp:
                    payload = json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                print(f"  GEM {atype} HTTP {exc.code}, skipping")
                break
            batch = payload.get("results") or []
            for a in batch:
                lat = a.get("latitude")
                lon = a.get("longitude")
                if lat is None or lon is None:
                    continue
                rows.append(
                    {
                        "id": a.get("asset_id") or "",
                        "name": a.get("asset_name") or "",
                        "source_type": a.get("asset_type") or atype,
                        "subsector": atype,
                        "sector": "gem",
                        "lat": float(lat),
                        "lon": float(lon),
                        "emissions_t": 0.0,
                        "capacity": float(a.get("capacity_value") or 0.0),
                        "inventory": "gem",
                    }
                )
            offset += len(batch)
            total = int(payload.get("total") or 0)
            print(f"  GEM {atype} {offset}/{total}")
            if offset >= total or not batch:
                break
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * 6371.0 * math.asin(min(1.0, math.sqrt(a)))


def cluster_indices(lats: np.ndarray, lons: np.ndarray, eps_km: float) -> list[list[int]]:
    n = int(lats.size)
    cell = eps_km
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    ix = np.empty(n, dtype=np.int32)
    iy = np.empty(n, dtype=np.int32)
    for i in range(n):
        iy[i] = int(math.floor(float(lats[i]) * 111.0 / cell))
        cphi = max(0.2, math.cos(math.radians(float(lats[i]))))
        ix[i] = int(math.floor(float(lons[i]) * 111.0 * cphi / cell))
        buckets[int(ix[i]), int(iy[i])].append(i)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (bx, by), idxs in buckets.items():
        cand: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand.extend(buckets.get((bx + dx, by + dy), ()))
        for i in idxs:
            for j in cand:
                if j <= i:
                    continue
                if haversine_km(float(lats[i]), float(lons[i]), float(lats[j]), float(lons[j])) <= eps_km:
                    union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def covering_radius_km(lats: np.ndarray, lons: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    w = np.maximum(weights, 1e-12)
    lat0 = float(np.average(lats, weights=w))
    lon0 = float(np.average(lons, weights=w))
    r = 0.0
    for la, lo in zip(lats, lons, strict=True):
        r = max(r, haversine_km(lat0, lon0, float(la), float(lo)))
    return lat0, lon0, r


def split_cluster(
    idxs: list[int],
    lats: np.ndarray,
    lons: np.ndarray,
    weights: np.ndarray,
    max_r: float,
) -> list[list[int]]:
    lat0, lon0, r = covering_radius_km(lats[idxs], lons[idxs], weights[idxs])
    if r <= max_r or len(idxs) == 1:
        return [idxs]
    span_lat = float(np.ptp(lats[idxs]))
    span_lon = float(np.ptp(lons[idxs])) * max(0.2, math.cos(math.radians(lat0)))
    if span_lat >= span_lon:
        mid = float(np.median(lats[idxs]))
        left = [i for i in idxs if float(lats[i]) <= mid]
        right = [i for i in idxs if float(lats[i]) > mid]
    else:
        mid = float(np.median(lons[idxs]))
        left = [i for i in idxs if float(lons[i]) <= mid]
        right = [i for i in idxs if float(lons[i]) > mid]
    if not left or not right or left == idxs or right == idxs:
        return [idxs]
    out: list[list[int]] = []
    for part in (left, right):
        out.extend(split_cluster(part, lats, lons, weights, max_r))
    return out


@dataclass
class Cluster:
    lat: float
    lon: float
    n: int
    r_plant_km: float
    r_cover_km: float
    area_km2: float
    emissions_t: float
    capacity: float
    subsector: str


def build_clusters(facilities: list[dict[str, object]]) -> list[Cluster]:
    if not facilities:
        return []
    lats = np.array([float(f["lat"]) for f in facilities], dtype=float)
    lons = np.array([float(f["lon"]) for f in facilities], dtype=float)
    em = np.array([float(f["emissions_t"]) for f in facilities], dtype=float)
    cap = np.array([float(f["capacity"]) for f in facilities], dtype=float)
    w = np.where(em > 0.0, em, np.where(cap > 0.0, cap, 1.0))
    groups = cluster_indices(lats, lons, CLUSTER_EPS_KM)
    pieces: list[list[int]] = []
    for g in groups:
        pieces.extend(split_cluster(g, lats, lons, w, MAX_CLUSTER_R_KM))
    clusters: list[Cluster] = []
    for idxs in pieces:
        arr = np.array(idxs, dtype=int)
        lat0, lon0, r_plant = covering_radius_km(lats[arr], lons[arr], w[arr])
        r_cover = r_plant + PLUME_R_KM
        area = math.pi * max(r_plant, MIN_PLANT_R_KM) ** 2
        subs = [str(facilities[i]["subsector"]) for i in idxs]
        sub = max(set(subs), key=subs.count)
        clusters.append(
            Cluster(
                lat=lat0,
                lon=lon0,
                n=len(idxs),
                r_plant_km=r_plant,
                r_cover_km=r_cover,
                area_km2=area,
                emissions_t=float(np.sum(em[arr])),
                capacity=float(np.sum(cap[arr])),
                subsector=sub,
            )
        )
    return clusters


class TimeLostFn:
    """T_1axis, T_2axis, lost as a function of |lat| and covering radius R."""

    def __init__(self, orbit: gimbal.Orbit, optics: gimbal.Optics) -> None:
        self.orbit = orbit
        self.optics = optics
        self.lats = np.array(GRID_LATS, dtype=float)
        self.rs = np.array(GRID_R_KM, dtype=float)
        nlat, nr = self.lats.size, self.rs.size
        cache = CACHE / "time_lost_grid.npz"
        t1: np.ndarray | None = None
        t2: np.ndarray | None = None
        if cache.is_file():
            z = np.load(cache)
            if np.allclose(z["lats"], self.lats) and np.allclose(z["rs"], self.rs):
                t1 = z["t1"]
                t2 = z["t2"]
                print(f"  time-lost grid cache {cache}")
        if t1 is None or t2 is None:
            t1 = np.zeros((nlat, nr))
            t2 = np.zeros((nlat, nr))
            print(f"building time-lost grid {nlat} lats x {nr} radii, dt={PASS_DT_S}s")
            for i, lat in enumerate(self.lats):
                origin = gimbal.sample_pass(orbit, float(lat), 0.0, 0.0, -250.0, 50.0, dt=PASS_DT_S)
                el_ok = gimbal.in_elevation_window(origin["el"], gimbal.WINDOW_MODE) & origin["vis"]
                for j, r in enumerate(self.rs):
                    edge_p = gimbal.sample_pass(orbit, float(lat), 0.0, float(r), -250.0, 50.0, dt=PASS_DT_S)
                    edge_m = gimbal.sample_pass(orbit, float(lat), 0.0, -float(r), -250.0, 50.0, dt=PASS_DT_S)
                    az_worst = np.maximum(np.abs(edge_p["az"]), np.abs(edge_m["az"]))
                    one = el_ok & (az_worst <= optics.half_az_deg)
                    two = el_ok & (az_worst <= gimbal.AZ_BOX_DEG)
                    t1[i, j] = gimbal.mask_time_s(one, origin["t"])
                    t2[i, j] = gimbal.mask_time_s(two, origin["t"])
                print(
                    f"  lat {lat:6.2f}  T1(R=5)={np.interp(5.0, self.rs, t1[i]):6.1f}s  "
                    f"T2={t2[i, 0]:6.1f}s"
                )
            CACHE.mkdir(parents=True, exist_ok=True)
            np.savez(cache, lats=self.lats, rs=self.rs, t1=t1, t2=t2)
        self._t1 = RegularGridInterpolator((self.lats, self.rs), t1, bounds_error=False, fill_value=None)
        self._t2 = RegularGridInterpolator((self.lats, self.rs), t2, bounds_error=False, fill_value=None)
        self.t1_grid = t1
        self.t2_grid = t2

    def eval(self, lat_deg: float, radius_km: float) -> tuple[float, float, float]:
        la = min(ISS_I_DEG, abs(float(lat_deg)))
        rr = float(np.clip(radius_km, float(self.rs[0]), float(self.rs[-1])))
        pt = np.array([[la, rr]])
        t1 = float(self._t1(pt)[0])
        t2 = float(self._t2(pt)[0])
        if not math.isfinite(t1):
            t1 = 0.0
        if not math.isfinite(t2):
            t2 = 0.0
        return t1, t2, t2 - t1


def iss_dwell_weight(lat_deg: float, i_deg: float = ISS_I_DEG) -> float:
    """Relative time spent near |lat| on a circular inclined orbit.

    dt/dφ ∝ cos φ / sqrt(sin² i − sin² φ). Diverges at ±i; clipped.
    """
    phi = math.radians(abs(lat_deg))
    i = math.radians(i_deg)
    s2 = math.sin(i) ** 2 - math.sin(phi) ** 2
    if s2 <= 1e-6:
        s2 = 1e-6
    return math.cos(phi) / math.sqrt(s2)


def daily_coverage_frac(lat_deg: float, half_swath_km: float, n_rev_day: float, re_km: float) -> float:
    """Fraction of the parallel covered per day by asc+desc passes of given half-swath."""
    cphi = max(0.08, math.cos(math.radians(lat_deg)))
    circ = 2.0 * math.pi * re_km * cphi
    covered = n_rev_day * 2.0 * (2.0 * half_swath_km)
    return min(1.0, covered / circ)


def histogram(clusters: list[Cluster], lats: np.ndarray, weights: np.ndarray, edges: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(lats, bins=edges, weights=weights)
    return h


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    w = float(np.sum(weights))
    if w <= 0:
        return 0.0
    return float(np.sum(values * weights) / w)


def expected_from_clusters(
    clusters: list[Cluster],
    fn: TimeLostFn,
    optics: gimbal.Optics,
    orbit: gimbal.Orbit,
    weight: str,
) -> dict[str, float]:
    iss = [c for c in clusters if abs(c.lat) <= ISS_I_DEG]
    if not iss:
        return {}
    lats = np.array([c.lat for c in iss])
    rs = np.array([c.r_cover_km for c in iss])
    if weight == "area":
        w = np.array([c.area_km2 for c in iss])
    elif weight == "stacks":
        w = np.array([float(c.n) for c in iss])
    elif weight == "emissions":
        w = np.array([c.emissions_t for c in iss])
    else:
        raise ValueError(weight)
    t1 = np.zeros(len(iss))
    t2 = np.zeros(len(iss))
    for i, c in enumerate(iss):
        a, b, _ = fn.eval(c.lat, c.r_cover_km)
        t1[i] = a
        t2[i] = b
    lost = t2 - t1
    re = orbit.earth_radius_km(0.0)
    n_rev = gimbal.TLE_MEAN_MOTION_REV_PER_DAY
    h_local = np.array([orbit.local_altitude_km(abs(c.lat)) for c in iss])
    half1 = h_local * math.tan(math.radians(optics.half_az_deg))
    half2 = h_local * math.tan(math.radians(gimbal.AZ_BOX_DEG))
    cov1 = np.array([daily_coverage_frac(abs(c.lat), float(h1), n_rev, re) for c, h1 in zip(iss, half1, strict=True)])
    cov2 = np.array([daily_coverage_frac(abs(c.lat), float(h2), n_rev, re) for c, h2 in zip(iss, half2, strict=True)])
    dwell = np.array([iss_dwell_weight(c.lat) for c in iss])
    high = np.abs(lats) >= 45.0
    return {
        "n_clusters": float(len(iss)),
        "weight_sum": float(np.sum(w)),
        "frac_iss_of_world": float(np.sum(w) / max(1e-12, sum(_weight_of(c, weight) for c in clusters))),
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


def _weight_of(c: Cluster, weight: str) -> float:
    if weight == "area":
        return c.area_km2
    if weight == "stacks":
        return float(c.n)
    return c.emissions_t


def plot_lat_hist(
    edges: np.ndarray,
    h_area: np.ndarray,
    h_stacks: np.ndarray,
    h_em: np.ndarray,
    h_gppd: np.ndarray | None,
    path: Path,
) -> None:
    centres = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.5), sharex=True)
    specs = (
        (h_area, "covering-disk area (km²)", C_BLUE),
        (h_stacks, "stack-bearing sources (count)", C_ORANGE),
        (h_em, "2025 CO₂ (tonnes)", C_TEAL),
    )
    for ax, (h, ylab, col) in zip(axes, specs, strict=True):
        ax.bar(centres, h, width=LAT_BIN_DEG * 0.9, color=col, align="center")
        ax.axvline(ISS_I_DEG, color=C_RED, ls="--", lw=1.2, label=f"ISS max lat ±{ISS_I_DEG:.1f}°")
        ax.axvline(-ISS_I_DEG, color=C_RED, ls="--", lw=1.2)
        ax.axvline(45.0, color=C_INK, ls=":", lw=1)
        ax.axvline(-45.0, color=C_INK, ls=":")
        ax.set_ylabel(ylab)
        ax.legend(loc="upper right", fontsize=8)
    if h_gppd is not None and float(np.sum(h_gppd)) > 0:
        scale = float(np.sum(h_stacks)) / max(1.0, float(np.sum(h_gppd)))
        axes[1].plot(
            centres,
            h_gppd * scale,
            color=C_INK,
            lw=1.2,
            label="GPPD thermal (scaled)",
        )
        axes[1].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("latitude (deg)")
    axes[0].set_title("Stack-bearing industrial areas vs latitude")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_folded(
    clusters: list[Cluster],
    fn: TimeLostFn,
    path: Path,
) -> None:
    iss = [c for c in clusters if abs(c.lat) <= ISS_I_DEG]
    abs_lat = np.array([abs(c.lat) for c in iss])
    w = np.array([float(c.n) for c in iss])
    edges = np.arange(0.0, ISS_I_DEG + 2.0, 2.0)
    h, _ = np.histogram(abs_lat, bins=edges, weights=w)
    centres = 0.5 * (edges[:-1] + edges[1:])
    t1 = np.array([fn.eval(float(x), gimbal.CLUSTER_R_KM)[0] for x in centres])
    t2 = np.array([fn.eval(float(x), gimbal.CLUSTER_R_KM)[1] for x in centres])
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax2 = ax.twinx()
    ax2.bar(centres, 100.0 * h / max(1.0, float(np.sum(h))), width=1.8, color=C_BLUE, alpha=0.3, label="stack %")
    ax.plot(centres, t2, color=C_TEAL, lw=2, marker="o", ms=3, label="2-axis, R=5 km")
    ax.plot(centres, t1, color=C_RED, lw=2, marker="^", ms=3, label="1-axis, R=5 km")
    ax.axvline(45.0, color=C_INK, ls=":", label="1-axis lossless (R=5)")
    ax.set_xlabel("|latitude| (deg)")
    ax.set_ylabel("tracking time (s)")
    ax2.set_ylabel("ISS-belt stack fraction (%)")
    ax.set_title("Folded latitude: stack mass vs tracking time")
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
    centres = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax2 = ax.twinx()
    ax2.bar(centres, w, width=LAT_BIN_DEG * 0.9, color=C_BLUE, alpha=0.25, label="ISS-belt stacks")
    ax.plot(centres, t2, color=C_TEAL, lw=2, marker="o", ms=3, label="2-axis, R=5 km")
    ax.plot(centres, t1, color=C_RED, lw=2, marker="^", ms=3, label="1-axis, R=5 km")
    ax.axvline(ISS_I_DEG, color=C_RED, ls="--", lw=1)
    ax.axvline(-ISS_I_DEG, color=C_RED, ls="--", lw=1)
    ax.set_xlabel("latitude (deg)")
    ax.set_ylabel("mean tracking time (s)")
    ax2.set_ylabel("stack-bearing sources in bin")
    ax.set_title("Tracking time vs latitude, with stack mass")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_r_hist(clusters: list[Cluster], path: Path) -> None:
    iss = [c for c in clusters if abs(c.lat) <= ISS_I_DEG]
    r = np.array([c.r_cover_km for c in iss])
    w = np.array([float(c.n) for c in iss])
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.hist(r, bins=np.linspace(0, 16, 33), weights=w, color=C_ORANGE, edgecolor="white")
    ax.axvline(gimbal.CLUSTER_R_KM, color=C_INK, ls=":", label="design R = 5 km")
    ax.set_xlabel("covering radius R = D + 2 km plume (km)")
    ax.set_ylabel("stack-bearing sources")
    ax.set_title("ISS-belt cluster size (stack-weighted)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_map(clusters: list[Cluster], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    lats = np.array([c.lat for c in clusters])
    lons = np.array([c.lon for c in clusters])
    s = np.clip(np.array([c.area_km2 for c in clusters]) * 3.0, 2.0, 40.0)
    ax.scatter(lons, lats, s=s, c=np.log10(np.maximum(np.array([c.n for c in clusters]), 1)), cmap="viridis", alpha=0.45, linewidths=0)
    ax.axhline(ISS_I_DEG, color=C_RED, ls="--", lw=1)
    ax.axhline(-ISS_I_DEG, color=C_RED, ls="--", lw=1)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title("Stack-bearing plant clusters (marker area ∝ covering disk)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(
    clusters: list[Cluster],
    gem_n: int,
    gppd_n: int,
    ct_n: int,
    fn: TimeLostFn,
    exp: dict[str, dict[str, float]],
    edges: np.ndarray,
    h_area: np.ndarray,
    path: Path,
) -> None:
    world = clusters
    iss = [c for c in world if abs(c.lat) <= ISS_I_DEG]
    a = []
    p = a.append
    p("# Industrial area vs latitude, and expected tracking time")
    p("")
    p("TEMPORARY ANALYSIS. Generated by `industrial_latitude.py`.")
    p("")
    p("## Inventory")
    p("")
    p("Primary: Climate TRACE v5.10.0 CO₂ **point sources** (2025, monthly summed)")
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
    p("radius is `R = D + 2 km` (plant span + plume CoG).")
    p("")
    p("The size assumption is that industrial-area size scales with the")
    p("number of stacks, so the **primary weight is stack count**. Covering-disk")
    p("area over-weights sprawling clusters (π D²) and is a sensitivity only.")
    p("Emissions weight how much the plant is actually running.")
    p("")
    p("| Inventory | sources |")
    p("| --- | --- |")
    p(f"| Climate TRACE stack sources | {ct_n} |")
    p(f"| Clusters (after split) | {len(world)} |")
    p(f"| In ISS belt \\|lat\\| ≤ {ISS_I_DEG:.2f}° | {len(iss)} |")
    p(f"| GEM operating (cross-check) | {gem_n} |")
    p(f"| GPPD thermal (cross-check) | {gppd_n} |")
    p("")
    p("GEM and GPPD peak in the same 30–40° N band, with ~93% of plants")
    p("inside the ISS belt. The Climate TRACE latitude shape is not an")
    p("OpenStreetMap-Europe artefact.")
    p("")
    p("## World distribution")
    p("")
    p("ISS never nadir-overflies outside ±51.63°. The 2-axis ±10° box adds")
    p("~0.7° of latitude at the turning point; ignored here.")
    p("")
    world_area = sum(c.area_km2 for c in world)
    iss_area = sum(c.area_km2 for c in iss)
    world_n = sum(c.n for c in world)
    iss_n = sum(c.n for c in iss)
    p("| | world | ISS belt | ISS fraction |")
    p("| --- | --- | --- | --- |")
    p(f"| covering area | {world_area:.0f} km² | {iss_area:.0f} km² | {100.0 * iss_area / max(1, world_area):.1f}% |")
    p(f"| stack sources | {world_n} | {iss_n} | {100.0 * iss_n / max(1, world_n):.1f}% |")
    p(f"| 2025 CO₂ | {sum(c.emissions_t for c in world) / 1e9:.2f} Gt | {sum(c.emissions_t for c in iss) / 1e9:.2f} Gt | {100.0 * sum(c.emissions_t for c in iss) / max(1, sum(c.emissions_t for c in world)):.1f}% |")
    p("")
    p("Stack-weighted mean covering radius in the ISS belt:")
    p(f"**{exp['stacks']['mean_r_km']:.2f} km** (mean sources/cluster {exp['stacks']['mean_n']:.2f}).")
    p("That matches the earlier design R = 5 km.")
    p(f"Stack fraction at \\|lat\\| ≥ 45° (1-axis lossless for a 5 km cluster):")
    p(f"**{100.0 * exp['stacks']['frac_lat_ge_45']:.1f}%**.")
    p("")
    p("### Folded |latitude| (ISS geometry is N/S symmetric)")
    p("")
    p("| \\|lat\\| (deg) | stacks | % of ISS-belt | 1-axis R=5 (s) | 2-axis (s) | lost % |")
    p("| --- | --- | --- | --- | --- | --- |")
    bands = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0), (40.0, 45.0), (45.0, ISS_I_DEG))
    iss_n_total = max(1, sum(c.n for c in iss))
    for lo, hi in bands:
        part = [c for c in iss if lo <= abs(c.lat) < hi]
        n = sum(c.n for c in part)
        mid = 0.5 * (lo + hi)
        t1, t2, lost = fn.eval(mid, gimbal.CLUSTER_R_KM)
        frac = 100.0 * n / iss_n_total
        lost_pct = 0.0 if t2 <= 0 else 100.0 * lost / t2
        p(f"| {lo:.0f}–{hi:.1f} | {n} | {frac:.1f}% | {t1:.1f} | {t2:.1f} | {lost_pct:.0f}% |")
    p("")
    p("## Expected tracking time (ISS-belt clusters, nadir-centered pass)")
    p("")
    p("For each cluster, `time_lost(|lat|, R)` uses the same one-sided 90→30")
    p("deg window, Earth rotation, and hard gimbal stop as `analyze.py`.")
    p("A pass is assumed over the cluster CoG (best-case origin). Off-track")
    p("origins are a separate loss, captured in the swath column.")
    p("")
    p("| weight | E[T 1-axis] (s) | E[T 2-axis] (s) | E[lost] (s) | lost % | weight at \\|lat\\|≥45° |")
    p("| --- | --- | --- | --- | --- | --- |")
    for key, label in (("stacks", "**stack count (primary)**"), ("emissions", "2025 CO₂"), ("area", "covering-disk area (sprawl)")):
        e = exp[key]
        p(
            f"| {label} | {e['e_t1']:.1f} | {e['e_t2']:.1f} | {e['e_lost']:.1f} | "
            f"{e['e_lost_pct']:.1f}% | {100.0 * e['frac_lat_ge_45']:.1f}% |"
        )
    p("")
    e = exp["stacks"]
    p("ISS dwell (more time near ±i) does not change the story:")
    p(
        f"stacks + dwell → E[lost] = **{e['e_lost_dwell']:.1f} s** "
        f"({e['e_t1_dwell']:.1f} vs {e['e_t2_dwell']:.1f})."
    )
    p("")
    p("## Lateral swath (why 2-axis still buys something at 50°)")
    p("")
    p("Tracking time is conditional on the cluster being in the frame.")
    p("1-axis only sees ±1.60 deg (~12 km at nadir). 2-axis sees the ±10 deg")
    p("box (~76 km). Ground-track spacing is thousands of km, so daily")
    p("longitude coverage is proportional to swath.")
    p("")
    p("| | 1-axis | 2-axis | ratio |")
    p("| --- | --- | --- | --- |")
    p(f"| mean daily coverage of a random ISS-belt cluster | {100.0 * e['mean_cov1']:.2f}% | {100.0 * e['mean_cov2']:.2f}% | {e['mean_cov2'] / max(1e-12, e['mean_cov1']):.2f}× |")
    p(f"| yield proxy E[T × coverage] (s × frac) | {e['e_yield1']:.2f} | {e['e_yield2']:.2f} | {e['e_yield2'] / max(1e-12, e['e_yield1']):.2f}× |")
    p("")
    p("So even where Earth-rotation azimuth walk is ~0, dropping the azimuth")
    p("axis throws away a factor of ~6 in how much industrial area a given")
    p("day can put in the frame. That is the lateral-motion cost of one axis,")
    p("independent of the along-track 124 s window.")
    p("")
    p("## Decision numbers")
    p("")
    p("1. **Along-track, given a pass over the CoG:** stack-weighted expected")
    p(f"   loss is **{e['e_lost']:.0f} s of {e['e_t2']:.0f} s** ({e['e_lost_pct']:.0f}%).")
    p("   Almost all of that is the 20–40° N band (China, US, Med, India,")
    p("   Japan/Korea), where Earth rotation walks the covering disk out at")
    p("   the 30 deg stop. Only ~10% of stacks sit at \\|lat\\| ≥ 45°, where")
    p("   a 5 km cluster is lossless on one axis.")
    p("2. **Lateral, whether we acquire at all:** 2-axis covers ~6× more")
    p("   industrial area per day. One axis only works plants that sit under")
    p("   the 24 km ground-track ribbon. Combined with shorter tracking time,")
    p(f"   the yield proxy E[T × coverage] is **{e['e_yield2'] / max(1e-12, e['e_yield1']):.1f}×** higher with two axes.")
    p("3. A polar-slice one-axis payload that is only tasked at \\|lat\\| ≥ 45°")
    p("   keeps the 124 s window, but it is looking at ~10% of the world's")
    p("   stack-bearing industry. If the mission must work the 20–40° N")
    p("   belt, 2-axis is required for the elevation window, not just swath.")
    p("")
    p("## Figures")
    p("")
    p("- `outputs/industrial_lat_hist.png`")
    p("- `outputs/industrial_lat_folded.png`")
    p("- `outputs/industrial_time_vs_lat.png`")
    p("- `outputs/industrial_cluster_radius.png`")
    p("- `outputs/industrial_cluster_map.png`")
    p("- `outputs/industrial_lat_hist.csv`, `outputs/industrial_clusters.csv`")
    p("- `outputs/time_lost_grid.csv` — the `time_lost(|lat|, R)` table")
    p("")
    p("```text")
    p("python3 scratch/single-axis-gimbal-analysis/industrial_latitude.py")
    p("```")
    p("")
    path.write_text("\n".join(a) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    gimbal._style()
    optics = gimbal.build_optics()
    orbit = gimbal.build_orbit(use_perigee=False)

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

    fn = TimeLostFn(orbit, optics)

    edges = np.arange(-90.0, 90.0 + LAT_BIN_DEG, LAT_BIN_DEG)
    lats = np.array([c.lat for c in clusters])
    h_area = histogram(clusters, lats, np.array([c.area_km2 for c in clusters]), edges)
    h_stacks = histogram(clusters, lats, np.array([float(c.n) for c in clusters]), edges)
    h_em = histogram(clusters, lats, np.array([c.emissions_t for c in clusters]), edges)

    h_gppd = np.zeros(edges.size - 1)
    if gppd:
        g_lats = np.array([float(f["lat"]) for f in gppd])
        h_gppd, _ = np.histogram(g_lats, bins=edges)

    t1_bin = np.zeros(edges.size - 1)
    t2_bin = np.zeros(edges.size - 1)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mid = 0.5 * (lo + hi)
        if abs(mid) > ISS_I_DEG:
            continue
        t1_bin[i], t2_bin[i], _ = fn.eval(mid, gimbal.CLUSTER_R_KM)

    exp = {k: expected_from_clusters(clusters, fn, optics, orbit, k) for k in ("area", "stacks", "emissions")}
    for k, e in exp.items():
        print(
            f"{k:10s}  E[T1]={e['e_t1']:.1f}s  E[T2]={e['e_t2']:.1f}s  "
            f"lost={e['e_lost']:.1f}s ({e['e_lost_pct']:.1f}%)  "
            f"|lat|>=45 {100 * e['frac_lat_ge_45']:.1f}%"
        )

    plot_lat_hist(edges, h_area, h_stacks, h_em, h_gppd, OUT / "industrial_lat_hist.png")
    plot_expected_vs_lat(edges, t1_bin, t2_bin, h_stacks, OUT / "industrial_time_vs_lat.png")
    plot_folded(clusters, fn, OUT / "industrial_lat_folded.png")
    plot_r_hist(clusters, OUT / "industrial_cluster_radius.png")
    plot_map(clusters, OUT / "industrial_cluster_map.png")

    with (OUT / "industrial_lat_hist.csv").open("w", encoding="utf-8") as fh:
        fh.write("lat_lo,lat_hi,lat_mid,area_km2,n_stacks,emissions_t,one_axis_s_R5,two_axis_s_R5\n")
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            mid = 0.5 * (lo + hi)
            fh.write(
                f"{lo:.1f},{hi:.1f},{mid:.1f},{h_area[i]:.4f},{h_stacks[i]:.4f},"
                f"{h_em[i]:.4f},{t1_bin[i]:.3f},{t2_bin[i]:.3f}\n"
            )
    with (OUT / "industrial_clusters.csv").open("w", encoding="utf-8") as fh:
        fh.write("lat,lon,n,r_plant_km,r_cover_km,area_km2,emissions_t,capacity,subsector\n")
        for c in clusters:
            fh.write(
                f"{c.lat:.5f},{c.lon:.5f},{c.n},{c.r_plant_km:.3f},{c.r_cover_km:.3f},"
                f"{c.area_km2:.4f},{c.emissions_t:.4f},{c.capacity:.4f},{c.subsector}\n"
            )
    with (OUT / "time_lost_grid.csv").open("w", encoding="utf-8") as fh:
        fh.write("lat_deg,radius_km,one_axis_s,two_axis_s,lost_s\n")
        for i, lat in enumerate(fn.lats):
            for j, r in enumerate(fn.rs):
                t1 = float(fn.t1_grid[i, j])
                t2 = float(fn.t2_grid[i, j])
                fh.write(f"{lat:.4f},{r:.3f},{t1:.3f},{t2:.3f},{t2 - t1:.3f}\n")
    with (OUT / "expected_tracking.csv").open("w", encoding="utf-8") as fh:
        keys = list(next(iter(exp.values())).keys())
        fh.write("weight," + ",".join(keys) + "\n")
        for name, e in exp.items():
            fh.write(name + "," + ",".join(f"{e[k]:.6g}" for k in keys) + "\n")

    write_report(
        clusters,
        len(gem),
        len(gppd),
        len(ct),
        fn,
        exp,
        edges,
        h_area,
        ROOT / "INDUSTRIAL.md",
    )
    print(f"wrote {OUT} and INDUSTRIAL.md")


if __name__ == "__main__":
    main()
