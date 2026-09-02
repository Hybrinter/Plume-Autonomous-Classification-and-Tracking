"""Stack-bearing facility inventories and plant-cluster construction.

Primary inventory is Climate TRACE v5 CO2 point sources that have smoke
stacks. GEM operating assets and GPPD thermal plants are cross-checks.
Nearby sources within CLUSTER_EPS_KM are one cluster; clusters wider than
MAX_CLUSTER_R_KM are split so two plants farther apart than the chip are
two tracks.

Contains:
  - Facility / Cluster dataclasses.
  - load_climate_trace / load_gppd / fetch_gem / load_clusters_csv.
  - haversine_km / cluster_indices / covering_radius_km / build_clusters.
"""

from __future__ import annotations

import csv
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from analysis.lib.constants import MEAN_EARTH_RADIUS_KM
from analysis.studies.single_axis_vs_dual_axis_gimbal.assumptions import (
    CACHE_DIR,
    CLUSTER_EPS_KM,
    INVENTORY_YEAR,
    MAX_CLUSTER_R_KM,
    MIN_PLANT_R_KM,
    PLUME_R_KM,
    RAW_DIR,
)

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


@dataclass
class Facility:
    """One stack-bearing point source.

    Attributes:
        id: Inventory source id.
        name: Facility name.
        source_type: Fuel or plant type string from the inventory.
        subsector: Climate TRACE / GEM subsector.
        sector: Inventory sector label.
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        emissions_t: Annual CO2 in tonnes (0 if unknown).
        capacity: Capacity in inventory units (MW or similar).
        inventory: Inventory name (climate-trace, gppd, gem).
    """

    id: str
    name: str
    source_type: str
    subsector: str
    sector: str
    lat: float
    lon: float
    emissions_t: float
    capacity: float
    inventory: str


@dataclass
class Cluster:
    """One plant cluster after 8 km grouping and covering-radius split.

    Attributes:
        lat: Weighted-centroid latitude in degrees.
        lon: Weighted-centroid longitude in degrees.
        n: Source count (stack count).
        r_plant_km: Plant-span covering radius in kilometres.
        r_cover_km: r_plant_km + plume CoG radius.
        area_km2: Covering-disk area using max(r_plant, MIN_PLANT_R_KM).
        emissions_t: Summed 2025 CO2 in tonnes.
        capacity: Summed capacity.
        subsector: Modal subsector of member sources.
    """

    lat: float
    lon: float
    n: int
    r_plant_km: float
    r_cover_km: float
    area_km2: float
    emissions_t: float
    capacity: float
    subsector: str


def _is_stack_power(source_type: str) -> bool:
    """Return True if an electricity-generation source_type is combustion.

    Args:
        source_type: Climate TRACE source_type string.

    Returns:
        False for hydro/solar/wind/nuclear and other non-stack tokens.
    """
    text = source_type.lower()
    if not text:
        return True
    return not any(tok in text for tok in NON_STACK_TOKENS)


def _ffloat(text: str) -> float | None:
    """Parse a float from a CSV cell, or None if empty/invalid.

    Args:
        text: Cell string.

    Returns:
        Parsed float, or None.
    """
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _json_float(value: object) -> float:
    """Parse a JSON number stored as int, float, or numeric string.

    Args:
        value: JSON scalar.

    Returns:
        float(value).

    Raises:
        TypeError: If value is not a number or numeric string.
    """
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise TypeError(f"not a number: {value!r}")
    return float(value)


def _facility_from_dict(row: dict[str, object]) -> Facility:
    """Build a Facility from a JSON-cache dict.

    Args:
        row: Dict with Facility field names.

    Returns:
        Facility.
    """
    return Facility(
        id=str(row["id"]),
        name=str(row["name"]),
        source_type=str(row["source_type"]),
        subsector=str(row["subsector"]),
        sector=str(row["sector"]),
        lat=_json_float(row["lat"]),
        lon=_json_float(row["lon"]),
        emissions_t=_json_float(row["emissions_t"]),
        capacity=_json_float(row["capacity"]),
        inventory=str(row["inventory"]),
    )


def load_ct_file(path: Path) -> list[Facility]:
    """Load one Climate TRACE emissions_sources CSV for INVENTORY_YEAR CO2.

    Args:
        path: ``*_emissions_sources_v5_10_0.csv`` path.

    Returns:
        One Facility per source_id (monthly rows summed).

    Raises:
        KeyError: If a required column is missing.
    """
    acc: dict[str, Facility] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        need = (
            "source_id",
            "source_name",
            "source_type",
            "subsector",
            "start_time",
            "lat",
            "lon",
            "gas",
            "emissions_quantity",
            "capacity",
        )
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
        for row in reader:
            if row[i_gas] != "co2" or not row[i_start].startswith(INVENTORY_YEAR):
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
            em_t = _ffloat(row[i_em]) or 0.0
            cap = _ffloat(row[i_cap]) or 0.0
            sid = row[i_id]
            rec = acc.get(sid)
            if rec is None:
                acc[sid] = Facility(
                    id=sid,
                    name=row[i_name],
                    source_type=row[i_type],
                    subsector=sub,
                    sector=row[i_sec] if i_sec >= 0 else "",
                    lat=lat,
                    lon=lon,
                    emissions_t=em_t,
                    capacity=cap,
                    inventory="climate-trace",
                )
            else:
                rec.emissions_t += em_t
                if cap > rec.capacity:
                    rec.capacity = cap
    return list(acc.values())


def load_climate_trace() -> list[Facility]:
    """Load Climate TRACE stack sources, using the JSON cache if present.

    Returns:
        Climate TRACE facilities.

    Raises:
        FileNotFoundError: If no CSVs exist under RAW_DIR and there is no cache.
    """
    cache = CACHE_DIR / "ct_facilities.json"
    if cache.is_file():
        print(f"  CT cache {cache}")
        raw_rows = json.loads(cache.read_text(encoding="utf-8"))
        return [_facility_from_dict(r) for r in raw_rows]
    files: list[Path] = []
    for folder in (RAW_DIR / "ct_power", RAW_DIR / "ct_manufacturing", RAW_DIR / "ct_fossil"):
        data = folder / "DATA"
        if not data.is_dir():
            continue
        files.extend(sorted(data.glob("*_emissions_sources_v5_10_0.csv")))
    if not files:
        raise FileNotFoundError(f"no Climate TRACE CSVs under {RAW_DIR}")
    rows: list[Facility] = []
    for path in files:
        print(f"  CT {path.name} ...", flush=True)
        part = load_ct_file(path)
        print(f"    {len(part)} sources")
        rows.extend(part)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([asdict(f) for f in rows]), encoding="utf-8")
    return rows


def load_gppd() -> list[Facility]:
    """Load GPPD thermal plants if the CSV is present.

    Returns:
        GPPD facilities, or an empty list if the file is missing.
    """
    path = RAW_DIR / "gppd" / "global_power_plant_database.csv"
    if not path.is_file():
        return []
    rows: list[Facility] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fuel = (row.get("primary_fuel") or "").strip()
            if fuel not in GPPD_STACK_FUELS:
                continue
            lat = _ffloat(row.get("latitude") or "")
            lon = _ffloat(row.get("longitude") or "")
            cap = _ffloat(row.get("capacity_mw") or "")
            if lat is None or lon is None or cap is None or cap <= 0:
                continue
            rows.append(
                Facility(
                    id=row.get("gppd_idnr") or "",
                    name=row.get("name") or "",
                    source_type=fuel,
                    subsector="electricity-generation",
                    sector="power",
                    lat=lat,
                    lon=lon,
                    emissions_t=0.0,
                    capacity=cap,
                    inventory="gppd",
                )
            )
    return rows


def fetch_gem() -> list[Facility]:
    """Fetch GEM operating stack-type assets, using the JSON cache if present.

    Returns:
        GEM facilities. Partial results if an asset type returns HTTP error.
    """
    cache = CACHE_DIR / "gem_operating.json"
    if cache.is_file():
        raw_rows = json.loads(cache.read_text(encoding="utf-8"))
        return [_facility_from_dict(r) for r in raw_rows]
    rows: list[Facility] = []
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "PACT-analysis/1.0")]
    for atype in GEM_STACK_TYPES:
        offset = 0
        limit = 500
        while True:
            url = "https://api.globalenergymonitor.org/assets?" + urllib.parse.urlencode(
                {
                    "asset_type": atype,
                    "status": "operating",
                    "limit": limit,
                    "offset": offset,
                }
            )
            try:
                with opener.open(url, timeout=60) as resp:
                    payload = json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                print(f"  GEM {atype} HTTP {exc.code}, skipping")
                break
            batch = payload.get("results") or []
            for asset in batch:
                lat = asset.get("latitude")
                lon = asset.get("longitude")
                if lat is None or lon is None:
                    continue
                rows.append(
                    Facility(
                        id=str(asset.get("asset_id") or ""),
                        name=str(asset.get("asset_name") or ""),
                        source_type=str(asset.get("asset_type") or atype),
                        subsector=atype,
                        sector="gem",
                        lat=float(lat),
                        lon=float(lon),
                        emissions_t=0.0,
                        capacity=float(asset.get("capacity_value") or 0.0),
                        inventory="gem",
                    )
                )
            offset += len(batch)
            total = int(payload.get("total") or 0)
            print(f"  GEM {atype} {offset}/{total}")
            if offset >= total or not batch:
                break
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([asdict(f) for f in rows]), encoding="utf-8")
    return rows


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometres on a spherical Earth.

    Args:
        lat1: First latitude in degrees.
        lon1: First longitude in degrees.
        lat2: Second latitude in degrees.
        lon2: Second longitude in degrees.

    Returns:
        Distance in kilometres.
    """
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlon = math.radians(lon2 - lon1)
    a_val = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * MEAN_EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a_val)))


def cluster_indices(lats: np.ndarray, lons: np.ndarray, eps_km: float) -> list[list[int]]:
    """Return connected components of sources within ``eps_km``.

    Args:
        lats: Latitudes in degrees. # np.ndarray[float64, (N,)]
        lons: Longitudes in degrees. # np.ndarray[float64, (N,)]
        eps_km: Link distance in kilometres.

    Returns:
        Lists of source indices, one list per connected component.
    """
    n_src = int(lats.size)
    cell = eps_km
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    ix = np.empty(n_src, dtype=np.int32)
    iy = np.empty(n_src, dtype=np.int32)
    for i in range(n_src):
        iy[i] = int(math.floor(float(lats[i]) * 111.0 / cell))
        cphi = max(0.2, math.cos(math.radians(float(lats[i]))))
        ix[i] = int(math.floor(float(lons[i]) * 111.0 * cphi / cell))
        buckets[int(ix[i]), int(iy[i])].append(i)
    parent = list(range(n_src))

    def find(a_idx: int) -> int:
        while parent[a_idx] != a_idx:
            parent[a_idx] = parent[parent[a_idx]]
            a_idx = parent[a_idx]
        return a_idx

    def union(a_idx: int, b_idx: int) -> None:
        ra, rb = find(a_idx), find(b_idx)
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
                if (
                    haversine_km(float(lats[i]), float(lons[i]), float(lats[j]), float(lons[j]))
                    <= eps_km
                ):
                    union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n_src):
        groups[find(i)].append(i)
    return list(groups.values())


def covering_radius_km(
    lats: np.ndarray, lons: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float]:
    """Return weighted centroid and max haversine radius of a point set.

    Args:
        lats: Latitudes in degrees. # np.ndarray[float64, (K,)]
        lons: Longitudes in degrees. # np.ndarray[float64, (K,)]
        weights: Non-negative weights. Zeros are floored to 1e-12.

    Returns:
        (lat0, lon0, radius_km).
    """
    w = np.maximum(weights, 1e-12)
    lat0 = float(np.average(lats, weights=w))
    lon0 = float(np.average(lons, weights=w))
    radius = 0.0
    for la, lo in zip(lats, lons, strict=True):
        radius = max(radius, haversine_km(lat0, lon0, float(la), float(lo)))
    return lat0, lon0, radius


def split_cluster(
    idxs: list[int],
    lats: np.ndarray,
    lons: np.ndarray,
    weights: np.ndarray,
    max_r: float,
) -> list[list[int]]:
    """Recursively split a cluster if its covering radius exceeds ``max_r``.

    Args:
        idxs: Source indices in this piece.
        lats: All latitudes. # np.ndarray[float64, (N,)]
        lons: All longitudes. # np.ndarray[float64, (N,)]
        weights: All weights. # np.ndarray[float64, (N,)]
        max_r: Maximum covering radius in kilometres.

    Returns:
        One or more index lists, each with covering radius <= max_r when splitable.
    """
    lat0, lon0, radius = covering_radius_km(lats[idxs], lons[idxs], weights[idxs])
    if radius <= max_r or len(idxs) == 1:
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


def build_clusters(facilities: list[Facility]) -> list[Cluster]:
    """Group facilities into plant clusters and split over-wide groups.

    Args:
        facilities: Point sources.

    Returns:
        Clusters with covering radius R = D + PLUME_R_KM.
    """
    if not facilities:
        return []
    lats = np.array([f.lat for f in facilities], dtype=float)
    lons = np.array([f.lon for f in facilities], dtype=float)
    em_t = np.array([f.emissions_t for f in facilities], dtype=float)
    cap = np.array([f.capacity for f in facilities], dtype=float)
    weights = np.where(em_t > 0.0, em_t, np.where(cap > 0.0, cap, 1.0))
    groups = cluster_indices(lats, lons, CLUSTER_EPS_KM)
    pieces: list[list[int]] = []
    for group in groups:
        pieces.extend(split_cluster(group, lats, lons, weights, MAX_CLUSTER_R_KM))
    clusters: list[Cluster] = []
    for idxs in pieces:
        arr = np.array(idxs, dtype=int)
        lat0, lon0, r_plant = covering_radius_km(lats[arr], lons[arr], weights[arr])
        r_cover = r_plant + PLUME_R_KM
        area = math.pi * max(r_plant, MIN_PLANT_R_KM) ** 2
        subs = [facilities[i].subsector for i in idxs]
        sub = max(set(subs), key=subs.count)
        clusters.append(
            Cluster(
                lat=lat0,
                lon=lon0,
                n=len(idxs),
                r_plant_km=r_plant,
                r_cover_km=r_cover,
                area_km2=area,
                emissions_t=float(np.sum(em_t[arr])),
                capacity=float(np.sum(cap[arr])),
                subsector=sub,
            )
        )
    return clusters


def load_clusters_csv(path: Path) -> list[Cluster]:
    """Load clusters previously written to ``industrial_clusters.csv``.

    Args:
        path: CSV path with the industry-command header.

    Returns:
        One Cluster per row.

    Raises:
        FileNotFoundError: If the CSV is missing.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    clusters: list[Cluster] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            clusters.append(
                Cluster(
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    n=int(float(row["n"])),
                    r_plant_km=float(row["r_plant_km"]),
                    r_cover_km=float(row["r_cover_km"]),
                    area_km2=float(row["area_km2"]),
                    emissions_t=float(row["emissions_t"]),
                    capacity=float(row["capacity"]),
                    subsector=row["subsector"],
                )
            )
    return clusters
