#!/usr/bin/env python3
"""TEMPORARY ANALYSIS — not flight software.

Single-axis vs two-axis gimbal tracking time for an ISS polar-slice view.

Locked assumptions (Aiden, 2026-09-01 follow-up):
  one-sided 90→30 deg window; geocentric nadir (≤5 deg later);
  at most one plant cluster; plume anywhere in the frame is useful;
  current ISS TLE; most restrictive FOV; full-frame inference;
  Earth rotation + inclination; stop at gimbal limits.

    uv run python scratch/single-axis-gimbal-analysis/analyze.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# LOCKED ASSUMPTIONS
# ---------------------------------------------------------------------------

# Celestrak GP TLE, NORAD 25544, epoch 2026-09-01 (day 244.4985).
# 1 25544U 98067A   26244.49851261  .00003910  00000-0  79223-4 0  9993
# 2 25544  51.6312 282.3953 0005055  96.4740 263.6825 15.48958602583586
TLE_EPOCH = "2026-09-01"
TLE_INCLINATION_DEG = 51.6312
TLE_ECCENTRICITY = 0.0005055
TLE_MEAN_MOTION_REV_PER_DAY = 15.48958602

MU_KM3_S2 = 398600.4418
OMEGA_EARTH_RAD_S = 7.2921159e-5
WGS84_A_KM = 6378.137
WGS84_B_KM = 6356.752314245

# Design pass: ISS at max latitude (heading due east). Polar-most nadir.
DESIGN_LAT_DEG = TLE_INCLINATION_DEG
PASS_LATS_DEG = (0.0, 20.0, 30.0, 40.0, 45.0, 50.0, TLE_INCLINATION_DEG)

# Operational pointing box. Hard gimbal stops for clearance — no residual FOV.
AZ_BOX_DEG = 10.0
EL_NADIR_DEG = 90.0
EL_LIMB_DEG = 30.0
WINDOW_MODE = "one_sided"
NADIR_OFFSET_DEG = 0.0  # ≤5 deg possible; treated as a later correction

# Hardware: BFS-U3-50S5 (IMX264) + 150 mm athermal. Long axis = lateral.
PIXEL_UM = 3.45
N_LATERAL_PX = 2448
N_ALONG_PX = 2048
FOCAL_LENGTH_MM = 150.0
DATASHEET_2_3_WIDTH_MM = 8.8
DATASHEET_HFOV_2_3_DEG = 3.36
LENS_DISTORTION_PCT = 0.66  # applied as a shrink of usable FOV (most restrictive)

MAX_SLEW_DEG_S = 2.0

# Covering radius of one plant cluster (stack span D + plume CoG radius r).
# Design 5 km: large integrated complex (~4 km wide, e.g. Baytown) + plume CoG.
CLUSTER_R_KM = 5.0
DISK_RADII_KM = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0)
WIND_MPS = (0.0, 10.0, 20.0)
ORIGIN_OFFSETS_KM = (0.0, 5.0, 10.0, 20.0, 40.0, 70.0)

# Full-frame inference = full 2x2 band plane of the chip.
BAND_LATERAL_PX = N_LATERAL_PX // 2
BAND_ALONG_PX = N_ALONG_PX // 2
# Selected flight models at 256 px (experiments/results/README.md stage 3).
SEG_FLOPS_256_G = 0.21
CLS_FLOPS_256_G = 0.110
SEG_LAT_256_MS = 2.3
CLS_LAT_256_MS = 3.4

DT_S = 0.05

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

C_BLUE = "#0077BB"
C_ORANGE = "#EE7733"
C_TEAL = "#009988"
C_RED = "#CC3311"
C_SKY = "#33BBEE"
C_INK = "#222222"


# ---------------------------------------------------------------------------
# Optics — most restrictive usable FOV
# ---------------------------------------------------------------------------


def fov_deg(n_px: int, pixel_um: float, fl_mm: float) -> float:
    half_mm = (n_px * pixel_um / 1000.0) / 2.0
    return 2.0 * math.degrees(math.atan(half_mm / fl_mm))


def ifov_deg_per_px(pixel_um: float, fl_mm: float) -> float:
    return math.degrees(math.atan((pixel_um / 1000.0) / fl_mm))


@dataclass(frozen=True)
class Optics:
    fov_az_raw_deg: float
    fov_el_raw_deg: float
    fov_az_deg: float
    fov_el_deg: float
    ifov_deg: float
    sensor_width_mm: float
    sensor_height_mm: float
    datasheet_hfov_deg: float
    computed_2_3_hfov_deg: float

    @property
    def half_az_deg(self) -> float:
        return 0.5 * self.fov_az_deg

    @property
    def half_el_deg(self) -> float:
        return 0.5 * self.fov_el_deg


def build_optics() -> Optics:
    raw_az = fov_deg(N_LATERAL_PX, PIXEL_UM, FOCAL_LENGTH_MM)
    raw_el = fov_deg(N_ALONG_PX, PIXEL_UM, FOCAL_LENGTH_MM)
    shrink = 1.0 - LENS_DISTORTION_PCT / 100.0
    computed_23 = 2.0 * math.degrees(math.atan((DATASHEET_2_3_WIDTH_MM / 2.0) / FOCAL_LENGTH_MM))
    return Optics(
        fov_az_raw_deg=raw_az,
        fov_el_raw_deg=raw_el,
        fov_az_deg=raw_az * shrink,
        fov_el_deg=raw_el * shrink,
        ifov_deg=ifov_deg_per_px(PIXEL_UM, FOCAL_LENGTH_MM),
        sensor_width_mm=N_LATERAL_PX * PIXEL_UM / 1000.0,
        sensor_height_mm=N_ALONG_PX * PIXEL_UM / 1000.0,
        datasheet_hfov_deg=DATASHEET_HFOV_2_3_DEG,
        computed_2_3_hfov_deg=computed_23,
    )


# ---------------------------------------------------------------------------
# Orbit from TLE
# ---------------------------------------------------------------------------


def wgs84_geocentric_radius_km(lat_rad: float) -> float:
    c, s = math.cos(lat_rad), math.sin(lat_rad)
    num = (WGS84_A_KM**2 * c) ** 2 + (WGS84_B_KM**2 * s) ** 2
    den = (WGS84_A_KM * c) ** 2 + (WGS84_B_KM * s) ** 2
    return math.sqrt(num / den)


@dataclass(frozen=True)
class Orbit:
    sma_km: float
    radius_km: float  # geocentric ISS radius used for this run (circular)
    inclination_rad: float
    eccentricity: float
    n_rad_s: float
    v_km_s: float
    period_s: float
    perigee_alt_eq_km: float
    apogee_alt_eq_km: float
    mean_alt_eq_km: float

    def local_altitude_km(self, lat_deg: float) -> float:
        return self.radius_km - wgs84_geocentric_radius_km(math.radians(lat_deg))

    def earth_radius_km(self, lat_deg: float) -> float:
        return wgs84_geocentric_radius_km(math.radians(lat_deg))


def build_orbit(*, use_perigee: bool = False) -> Orbit:
    n = TLE_MEAN_MOTION_REV_PER_DAY * 2.0 * math.pi / 86400.0
    a = (MU_KM3_S2 / n**2) ** (1.0 / 3.0)
    rp = a * (1.0 - TLE_ECCENTRICITY)
    ra = a * (1.0 + TLE_ECCENTRICITY)
    radius = rp if use_perigee else a
    v = math.sqrt(MU_KM3_S2 / radius)
    n_used = math.sqrt(MU_KM3_S2 / radius**3)
    return Orbit(
        sma_km=a,
        radius_km=radius,
        inclination_rad=math.radians(TLE_INCLINATION_DEG),
        eccentricity=TLE_ECCENTRICITY,
        n_rad_s=n_used,
        v_km_s=v,
        period_s=2.0 * math.pi / n_used,
        perigee_alt_eq_km=rp - WGS84_A_KM,
        apogee_alt_eq_km=ra - WGS84_A_KM,
        mean_alt_eq_km=a - WGS84_A_KM,
    )


def argument_of_latitude(lat_deg: float, inclination_rad: float) -> float:
    """Northern-hemisphere argument of latitude for a given geocentric latitude."""
    s = math.sin(math.radians(lat_deg)) / math.sin(inclination_rad)
    s = max(-1.0, min(1.0, s))
    return math.asin(s)


def heading_from_north_deg(lat_deg: float, inclination_rad: float) -> float:
    """Ground-track azimuth from north at the given latitude (ascending)."""
    cphi = math.cos(math.radians(lat_deg))
    if abs(cphi) < 1e-12:
        return 90.0
    s = min(1.0, math.cos(inclination_rad) / cphi)
    return math.degrees(math.asin(s))


def iss_eci(t_s: float, orbit: Orbit, u0: float) -> tuple[np.ndarray, np.ndarray]:
    """Circular-orbit ECI with node on the X axis, argp = 0."""
    u = u0 + orbit.n_rad_s * t_s
    i = orbit.inclination_rad
    a = orbit.radius_km
    r = a * np.array([math.cos(u), math.sin(u) * math.cos(i), math.sin(u) * math.sin(i)])
    drdu = a * np.array([-math.sin(u), math.cos(u) * math.cos(i), math.cos(u) * math.sin(i)])
    return r, orbit.n_rad_s * drdu


def rz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def origin_ecef(orbit: Orbit, lat_deg: float) -> np.ndarray:
    """Cluster CoG on the Earth surface, at the sub-satellite point at t = 0.

    ECI and ECEF are aligned at t = 0.
    """
    u0 = argument_of_latitude(lat_deg, orbit.inclination_rad)
    r0, _ = iss_eci(0.0, orbit, u0)
    re = orbit.earth_radius_km(lat_deg)
    return r0 / np.linalg.norm(r0) * re


def offset_ecef(
    origin: np.ndarray,
    lat_deg: float,
    along_km: float,
    cross_km: float,
    heading_deg: float,
) -> np.ndarray:
    """Offset a surface point in the along-track / cross-track tangent plane."""
    r = float(np.linalg.norm(origin))
    lat = math.radians(lat_deg)
    lon = math.atan2(float(origin[1]), float(origin[0]))
    north = np.array([-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)])
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    az = math.radians(heading_deg)
    along_hat = math.cos(az) * north + math.sin(az) * east
    cross_hat = -math.sin(az) * north + math.cos(az) * east
    vec = origin + along_hat * along_km + cross_hat * cross_km
    return vec / np.linalg.norm(vec) * r


def body_axes(r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = -r / np.linalg.norm(r)
    x = v - np.dot(v, z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    return x, y, z


@dataclass(frozen=True)
class Look:
    az_deg: float
    el_deg: float
    eta_deg: float
    slant_km: float
    visible: bool


def look_at(r_iss: np.ndarray, v: np.ndarray, target_eci: np.ndarray) -> Look:
    x, y, z = body_axes(r_iss, v)
    look = target_eci - r_iss
    slant = float(np.linalg.norm(look))
    if slant < 1e-9:
        return Look(0.0, 90.0, 0.0, 0.0, False)
    u = look / slant
    lx = float(np.dot(look, x))
    ly = float(np.dot(look, y))
    lz = float(np.dot(look, z))
    eta = math.degrees(math.acos(float(np.clip(np.dot(u, z), -1.0, 1.0))))
    along_off = math.degrees(math.atan2(lx, lz))
    el = EL_NADIR_DEG - along_off
    az = math.degrees(math.atan2(ly, math.hypot(lx, lz)))
    vis = _hits_earth(r_iss, u, float(np.linalg.norm(target_eci)))
    return Look(az_deg=az, el_deg=el, eta_deg=eta, slant_km=slant, visible=vis)


def _hits_earth(s: np.ndarray, u: np.ndarray, re: float) -> bool:
    b = 2.0 * float(np.dot(s, u))
    c = float(np.dot(s, s)) - re**2
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return False
    sqrt_d = math.sqrt(disc)
    t1 = (-b - sqrt_d) / 2.0
    t2 = (-b + sqrt_d) / 2.0
    return t1 > 0.0 or t2 > 0.0


def earth_hit(s: np.ndarray, u: np.ndarray, re: float) -> np.ndarray | None:
    b = 2.0 * float(np.dot(s, u))
    c = float(np.dot(s, s)) - re**2
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None
    sqrt_d = math.sqrt(disc)
    ts = [t for t in ((-b - sqrt_d) / 2.0, (-b + sqrt_d) / 2.0) if t > 1e-6]
    if not ts:
        return None
    return s + min(ts) * u


def sample_pass(
    orbit: Orbit,
    lat_deg: float,
    along_km: float,
    cross_km: float,
    t_min: float,
    t_max: float,
    dt: float = DT_S,
) -> dict[str, np.ndarray]:
    u0 = argument_of_latitude(lat_deg, orbit.inclination_rad)
    heading = heading_from_north_deg(lat_deg, orbit.inclination_rad)
    origin = origin_ecef(orbit, lat_deg)
    tgt_ecef = offset_ecef(origin, lat_deg, along_km, cross_km, heading)
    ts = np.arange(t_min, t_max + dt * 0.5, dt)
    n = ts.size
    az = np.empty(n)
    el = np.empty(n)
    eta = np.empty(n)
    slant = np.empty(n)
    vis = np.empty(n, dtype=bool)
    for i, t in enumerate(ts):
        r, v = iss_eci(float(t), orbit, u0)
        tgt = rz(OMEGA_EARTH_RAD_S * float(t)) @ tgt_ecef
        look = look_at(r, v, tgt)
        az[i] = look.az_deg
        el[i] = look.el_deg
        eta[i] = look.eta_deg
        slant[i] = look.slant_km
        vis[i] = look.visible
    return {"t": ts, "az": az, "el": el, "eta": eta, "slant": slant, "vis": vis}


def in_elevation_window(el_deg: np.ndarray, mode: str = WINDOW_MODE) -> np.ndarray:
    forward = (el_deg >= EL_LIMB_DEG) & (el_deg <= EL_NADIR_DEG)
    if mode == "one_sided":
        return forward
    return np.abs(el_deg - EL_NADIR_DEG) <= (EL_NADIR_DEG - EL_LIMB_DEG)


def mask_time_s(mask: np.ndarray, t: np.ndarray) -> float:
    if t.size < 2 or not np.any(mask):
        return 0.0
    dt = float(np.median(np.diff(t)))
    return float(np.sum(mask) * dt)


def angular_rate_deg_s(t: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    return np.gradient(angle_deg, t)


def two_axis_boresightable(az: np.ndarray, el: np.ndarray, mode: str) -> np.ndarray:
    return in_elevation_window(el, mode) & (np.abs(az) <= AZ_BOX_DEG)


def staring_in_frame(az: np.ndarray, el: np.ndarray, optics: Optics) -> np.ndarray:
    return (np.abs(az) <= optics.half_az_deg) & (np.abs(el - EL_NADIR_DEG) <= optics.half_el_deg)


def disk_points(radius_km: float, n: int = 72) -> np.ndarray:
    phis = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.stack([radius_km * np.cos(phis), radius_km * np.sin(phis)], axis=1)


def disk_max_az(
    orbit: Orbit,
    t: float,
    lat_deg: float,
    radius_km: float,
) -> float:
    u0 = argument_of_latitude(lat_deg, orbit.inclination_rad)
    heading = heading_from_north_deg(lat_deg, orbit.inclination_rad)
    origin = origin_ecef(orbit, lat_deg)
    r, v = iss_eci(t, orbit, u0)
    rot = rz(OMEGA_EARTH_RAD_S * t)
    worst = 0.0
    for along, cross in disk_points(radius_km):
        tgt = rot @ offset_ecef(origin, lat_deg, float(along), float(cross), heading)
        worst = max(worst, abs(look_at(r, v, tgt).az_deg))
    tgt0 = rot @ origin
    worst = max(worst, abs(look_at(r, v, tgt0).az_deg))
    return worst


@dataclass
class WindowTimes:
    along_track_s: float
    stare_s: float
    peak_el_rate_deg_s: float
    t_start_s: float
    t_stop_s: float
    slant_start_km: float
    slant_stop_km: float
    az_max_deg: float
    local_alt_km: float


def origin_window(
    orbit: Orbit,
    optics: Optics,
    lat_deg: float,
    origin_cross_km: float,
    mode: str,
) -> tuple[WindowTimes, dict[str, np.ndarray]]:
    data = sample_pass(orbit, lat_deg, 0.0, origin_cross_km, -250.0, 50.0)
    el_mask = in_elevation_window(data["el"], mode) & data["vis"]
    empty = WindowTimes(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if not np.any(el_mask):
        return empty, data
    t = data["t"]
    idx = np.where(el_mask)[0]
    t_start = float(t[idx[0]])
    t_stop = float(t[idx[-1]])
    stare = mask_time_s(staring_in_frame(data["az"], data["el"], optics), t)
    rate = np.abs(angular_rate_deg_s(t, data["el"]))
    peak_rate = float(np.max(rate[el_mask])) if np.any(el_mask) else 0.0
    return (
        WindowTimes(
            along_track_s=t_stop - t_start,
            stare_s=stare,
            peak_el_rate_deg_s=peak_rate,
            t_start_s=t_start,
            t_stop_s=t_stop,
            slant_start_km=float(data["slant"][idx[0]]),
            slant_stop_km=float(data["slant"][idx[-1]]),
            az_max_deg=float(np.max(np.abs(data["az"][el_mask]))),
            local_alt_km=orbit.local_altitude_km(lat_deg),
        ),
        data,
    )


def tracking_time_vs_radius(
    orbit: Orbit,
    optics: Optics,
    lat_deg: float,
    radii: tuple[float, ...],
    mode: str,
) -> dict[str, np.ndarray]:
    r_arr = np.array(radii, dtype=float)
    t1 = np.zeros(r_arr.size)
    t2 = np.zeros(r_arr.size)
    az_peak = np.zeros(r_arr.size)
    for i, r in enumerate(r_arr):
        edge_p = sample_pass(orbit, lat_deg, 0.0, r, -250.0, 50.0)
        edge_m = sample_pass(orbit, lat_deg, 0.0, -r, -250.0, 50.0)
        origin = sample_pass(orbit, lat_deg, 0.0, 0.0, -250.0, 50.0)
        el_ok = in_elevation_window(origin["el"], mode) & origin["vis"]
        az_worst = np.maximum(np.abs(edge_p["az"]), np.abs(edge_m["az"]))
        one = el_ok & (az_worst <= optics.half_az_deg)
        two = el_ok & (az_worst <= AZ_BOX_DEG)
        t1[i] = mask_time_s(one, origin["t"])
        t2[i] = mask_time_s(two, origin["t"])
        az_peak[i] = float(np.max(az_worst[el_ok])) if np.any(el_ok) else 0.0
    return {
        "radius_km": r_arr,
        "one_axis_s": t1,
        "two_axis_s": t2,
        "lost_s": t2 - t1,
        "az_peak_deg": az_peak,
    }


def latitude_table(
    orbit: Orbit,
    optics: Optics,
    lats: tuple[float, ...],
    radius_km: float,
    mode: str,
) -> list[tuple[float, float, float, float, float, float]]:
    """lat, h, el_window_s, az_max, one_axis_s, two_axis_s."""
    rows: list[tuple[float, float, float, float, float, float]] = []
    for lat in lats:
        times, origin = origin_window(orbit, optics, lat, 0.0, mode)
        edge_p = sample_pass(orbit, lat, 0.0, radius_km, -250.0, 50.0)
        edge_m = sample_pass(orbit, lat, 0.0, -radius_km, -250.0, 50.0)
        el_ok = in_elevation_window(origin["el"], mode) & origin["vis"]
        az_worst = np.maximum(np.abs(edge_p["az"]), np.abs(edge_m["az"]))
        one = el_ok & (az_worst <= optics.half_az_deg)
        two = el_ok & (az_worst <= AZ_BOX_DEG)
        rows.append(
            (
                lat,
                times.local_alt_km,
                times.along_track_s,
                times.az_max_deg,
                mask_time_s(one, origin["t"]),
                mask_time_s(two, origin["t"]),
            )
        )
    return rows


def offset_times(
    orbit: Orbit,
    optics: Optics,
    lat_deg: float,
    mode: str,
) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []
    for y in ORIGIN_OFFSETS_KM:
        _times, data = origin_window(orbit, optics, lat_deg, y, mode)
        el_ok = in_elevation_window(data["el"], mode) & data["vis"]
        one = el_ok & (np.abs(data["az"]) <= optics.half_az_deg)
        two = two_axis_boresightable(data["az"], data["el"], mode)
        rows.append((float(y), mask_time_s(one, data["t"]), mask_time_s(two, data["t"])))
    return rows


def footprint_half_width_km(
    orbit: Orbit,
    optics: Optics,
    t: float,
    lat_deg: float,
) -> tuple[float, float]:
    u0 = argument_of_latitude(lat_deg, orbit.inclination_rad)
    r, v = iss_eci(t, orbit, u0)
    origin = rz(OMEGA_EARTH_RAD_S * t) @ origin_ecef(orbit, lat_deg)
    re = float(np.linalg.norm(origin))
    x, y, z = body_axes(r, v)
    look0 = origin - r
    u0v = look0 / np.linalg.norm(look0)
    half_el = math.radians(optics.half_el_deg)
    half_az = math.radians(optics.half_az_deg)

    def rot(vec: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
        k = axis / np.linalg.norm(axis)
        return (
            vec * math.cos(ang)
            + np.cross(k, vec) * math.sin(ang)
            + k * np.dot(k, vec) * (1.0 - math.cos(ang))
        )

    p0 = earth_hit(r, u0v, re)
    p_el = earth_hit(r, rot(u0v, y, half_el), re)
    p_az = earth_hit(r, rot(u0v, x, half_az), re)
    if p0 is None or p_el is None or p_az is None:
        return float("nan"), float("nan")
    return float(np.linalg.norm(p_el - p0)), float(np.linalg.norm(p_az - p0))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "axes.edgecolor": C_INK,
            "text.color": C_INK,
            "axes.labelcolor": C_INK,
            "xtick.color": C_INK,
            "ytick.color": C_INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_along_track(
    orbit: Orbit, optics: Optics, times: WindowTimes, data: dict, lat_deg: float, path: Path
) -> None:
    t = data["t"]
    mask = (t >= times.t_start_s - 15) & (t <= times.t_stop_s + 15)
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.4), sharex=True)
    axes[0].plot(t[mask], data["el"][mask], color=C_BLUE, lw=2)
    axes[0].axhline(EL_NADIR_DEG, color=C_INK, ls=":", lw=1)
    axes[0].axhline(EL_LIMB_DEG, color=C_ORANGE, ls="--", lw=1, label="el stop 30 deg")
    axes[0].axvspan(times.t_start_s, times.t_stop_s, color=C_BLUE, alpha=0.12, label="elevation window")
    axes[0].set_ylabel("elevation (deg)")
    axes[0].set_ylim(20, 100)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title(
        f"Along-track track time = {times.along_track_s:.1f} s "
        f"(lat {lat_deg:.1f} deg, h = {times.local_alt_km:.0f} km, one-sided 90→30)"
    )

    rate = np.abs(angular_rate_deg_s(t, data["el"]))
    axes[1].plot(t[mask], rate[mask], color=C_ORANGE, lw=2)
    axes[1].axhline(MAX_SLEW_DEG_S, color=C_RED, ls="--", label=f"mission slew cap {MAX_SLEW_DEG_S:.0f} deg/s")
    axes[1].set_ylabel("|d(el)/dt| (deg/s)")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(t[mask], np.abs(data["az"][mask]), color=C_TEAL, lw=2, label="|az| of cluster origin")
    axes[2].axhline(optics.half_az_deg, color=C_INK, ls="--", label="sensor half-FOV")
    axes[2].set_ylabel("|az| (deg)")
    axes[2].set_xlabel("time from closest approach (s)")
    axes[2].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_disk_angle(orbit: Orbit, optics: Optics, times: WindowTimes, lat_deg: float, path: Path) -> None:
    t = np.arange(times.t_start_s, times.t_stop_s + 1.0, 1.0)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    radii = (1.0, 2.0, 5.0, 8.0, 10.0, 15.0)
    colors = [C_SKY, C_BLUE, C_TEAL, C_ORANGE, C_RED, "#882255"]
    for r, c in zip(radii, colors, strict=True):
        az = [disk_max_az(orbit, float(ti), lat_deg, r) for ti in t]
        ax.plot(t, az, color=c, lw=2, label=f"R = {r:.0f} km")
    ax.axhline(optics.half_az_deg, color=C_INK, ls="--", lw=1.4, label=f"sensor half-FOV {optics.half_az_deg:.2f} deg")
    ax.set_xlabel("time from closest approach (s)")
    ax.set_ylabel("worst-case |az| of covering disk (deg)")
    ax.set_title(f"Covering-disk azimuth at lat {lat_deg:.1f} deg (Earth rotation on)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_time_vs_radius(
    table: dict[str, np.ndarray], optics: Optics, h_km: float, lat_deg: float, path: Path
) -> None:
    r = table["radius_km"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(r, table["two_axis_s"], color=C_BLUE, lw=2.4, marker="o", label="two-axis (box ±10 deg az)")
    ax.plot(r, table["one_axis_s"], color=C_ORANGE, lw=2.4, marker="s", label="one-axis (az parked)")
    ax.axvline(
        h_km * math.tan(math.radians(optics.half_az_deg)),
        color=C_TEAL,
        ls="--",
        label="R = h tan(FOV_az / 2) at nadir",
    )
    ax.axvline(CLUSTER_R_KM, color=C_INK, ls=":", label=f"design cluster R = {CLUSTER_R_KM:.0f} km")
    ax.set_xlabel("covering-disk radius R (km)")
    ax.set_ylabel("in-frame tracking time (s)")
    ax.set_title(f"Tracking time vs covering radius at lat {lat_deg:.1f} deg")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_lost_time(table: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(table["radius_km"], table["lost_s"], color=C_RED, lw=2.4, marker="o")
    ax.axvline(CLUSTER_R_KM, color=C_INK, ls=":", label=f"design R = {CLUSTER_R_KM:.0f} km")
    ax.set_xlabel("covering-disk radius R (km)")
    ax.set_ylabel("tracking time lost vs two-axis (s)")
    ax.set_title("Time given up by dropping the azimuth axis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_footprint(orbit: Orbit, optics: Optics, times: WindowTimes, lat_deg: float, path: Path) -> None:
    t = np.linspace(times.t_start_s, times.t_stop_s, 80)
    along, cross = [], []
    for ti in t:
        a, c = footprint_half_width_km(orbit, optics, float(ti), lat_deg)
        along.append(a)
        cross.append(c)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(t, along, color=C_BLUE, lw=2, label="along-track half-footprint")
    ax.plot(t, cross, color=C_ORANGE, lw=2, label="cross-track half-footprint")
    ax.set_xlabel("time from closest approach (s)")
    ax.set_ylabel("ground half-width of sensor FOV (km)")
    ax.set_title(f"Sensor footprint at lat {lat_deg:.1f} deg")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_offset_map(rows: list[tuple[float, float, float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ys = [r[0] for r in rows]
    ax.plot(ys, [r[2] for r in rows], color=C_BLUE, lw=2.4, marker="o", label="two-axis")
    ax.plot(ys, [r[1] for r in rows], color=C_ORANGE, lw=2.4, marker="s", label="one-axis")
    ax.set_xlabel("cluster origin cross-track offset (km)")
    ax.set_ylabel("in-frame tracking time (s)")
    ax.set_title("If the cluster is not on the slice centerline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_required_az(table: dict[str, np.ndarray], optics: Optics, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(table["radius_km"], table["az_peak_deg"], color=C_BLUE, lw=2.4, marker="o")
    ax.axhline(optics.half_az_deg, color=C_ORANGE, ls="--", label="1-axis: sensor half-FOV")
    ax.axhline(AZ_BOX_DEG, color=C_INK, ls=":", label="2-axis operational az box")
    ax.set_xlabel("covering-disk radius R (km)")
    ax.set_ylabel("peak |az| during the elevation window (deg)")
    ax.set_title("Azimuth a 2-axis gimbal would use to keep the disk edge on boresight")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_latitude(
    rows0: list[tuple[float, float, float, float, float, float]],
    rows5: list[tuple[float, float, float, float, float, float]],
    optics: Optics,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.4), sharex=True)
    lat0 = [r[0] for r in rows0]
    axes[0].plot(lat0, [r[3] for r in rows0], color=C_TEAL, lw=2.4, marker="o", label="R = 0 (origin)")
    axes[0].axhline(optics.half_az_deg, color=C_INK, ls="--", label="sensor half-FOV")
    axes[0].set_ylabel("peak |az| of origin (deg)")
    axes[0].set_title("Earth rotation walks a nadir-centered origin in azimuth")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(lat0, [r[2] for r in rows0], color=C_BLUE, lw=2, marker="o", label="elevation window (2-axis)")
    axes[1].plot(lat0, [r[4] for r in rows0], color=C_ORANGE, lw=2, marker="s", label="1-axis, R = 0")
    axes[1].plot(
        [r[0] for r in rows5],
        [r[4] for r in rows5],
        color=C_RED,
        lw=2,
        marker="^",
        label=f"1-axis, R = {CLUSTER_R_KM:.0f} km cluster",
    )
    axes[1].set_xlabel("pass latitude (deg)")
    axes[1].set_ylabel("in-frame tracking time (s)")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(
    orbit: Orbit,
    optics: Optics,
    times: WindowTimes,
    table: dict[str, np.ndarray],
    offsets: list[tuple[float, float, float]],
    lat_rows_r0: list[tuple[float, float, float, float, float, float]],
    lat_rows_r5: list[tuple[float, float, float, float, float, float]],
    times_perigee: WindowTimes,
    path: Path,
) -> None:
    h = times.local_alt_km
    r_fit = h * math.tan(math.radians(optics.half_az_deg))
    r_box = h * math.tan(math.radians(AZ_BOX_DEG))
    area_256 = 256 * 256
    area_ff = BAND_LATERAL_PX * BAND_ALONG_PX
    scale = area_ff / area_256
    lines: list[str] = []
    a = lines.append
    a("# Single-axis gimbal tracking time")
    a("")
    a("TEMPORARY ANALYSIS. Generated by `analyze.py`. Not flight software.")
    a("")
    a("## Locked assumptions")
    a("")
    a("From the 2026-09-01 follow-up:")
    a("")
    a("- Elevation window is **one-sided** 90→30 deg. Hard gimbal stops for")
    a("  clearance; tracking time stops at the limit, with no leftover FOV walk-out.")
    a("- Mount is geocentric nadir. A ≤5 deg offset is a later correction, not in")
    a("  this run.")
    a("- At most **one plant cluster** per track. Elevation follows the cluster")
    a("  CoG. Science is satisfied if a plume is **anywhere in the frame**, so the")
    a("  success metric is a covering disk of radius R around that CoG, not")
    a("  boresight centering.")
    a("- If plants are spread farther than the chip (~±12 km at nadir), CoG is")
    a("  the wrong metric: pick the nearest/brightest cluster and treat it as")
    a("  its own origin. Do not average across a city.")
    a("- Current ISS TLE (Celestrak GP, epoch 2026-09-01). Earth rotation and")
    a("  51.63 deg inclination are on. Design pass is **max latitude** (heading")
    a("  due east), which is the polar-most nadir ISS can reach.")
    a("- FOV is the **most restrictive**: IMX264 active area, then shrunk by the")
    a("  lens max distortion of 0.66 %. The 8.8 mm 2/3\" sheet HFOV of 3.36 deg")
    a("  is larger than this camera and is not used.")
    a("- Full-frame inference = the full 2×2 band plane. That does not change")
    a("  optical FOV; it is a compute note.")
    a("- A ~2028 launch still sits in the current 416–423 km operational band.")
    a("  ISS is committed through 2030; retrograde deorbit lowering is a")
    a("  2028–2030 lead-up, not a 2028 operations baseline.")
    a("")
    a("## ISS orbit (TLE 25544, 2026-09-01)")
    a("")
    a("| Quantity | Value |")
    a("| --- | --- |")
    a(f"| Inclination | {TLE_INCLINATION_DEG:.4f} deg |")
    a(f"| Eccentricity | {TLE_ECCENTRICITY:.7f} |")
    a(f"| Mean motion | {TLE_MEAN_MOTION_REV_PER_DAY:.8f} rev/day |")
    a(f"| Period | {orbit.period_s / 60.0:.3f} min |")
    a(f"| SMA | {orbit.sma_km:.2f} km |")
    a(f"| Mean altitude (WGS84 equator) | {orbit.mean_alt_eq_km:.2f} km |")
    a(f"| Perigee / apogee altitude | {orbit.perigee_alt_eq_km:.2f} / {orbit.apogee_alt_eq_km:.2f} km |")
    a(f"| Inertial speed (SMA) | {orbit.v_km_s:.3f} km/s |")
    a(f"| Local altitude at lat {DESIGN_LAT_DEG:.2f} deg | **{h:.2f} km** |")
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
    a(f"| Array | {N_LATERAL_PX} × {N_ALONG_PX} (lateral × along-track) |")
    a(f"| Active area | {optics.sensor_width_mm:.3f} × {optics.sensor_height_mm:.3f} mm |")
    a(f"| Thin-lens FOV | {optics.fov_az_raw_deg:.3f} × {optics.fov_el_raw_deg:.3f} deg |")
    a(f"| Usable FOV after {LENS_DISTORTION_PCT:.2f} % distortion | **{optics.fov_az_deg:.3f} × {optics.fov_el_deg:.3f} deg** |")
    a(f"| Half-FOV | ±{optics.half_az_deg:.3f} deg az, ±{optics.half_el_deg:.3f} deg el |")
    a(f"| 2/3\" sheet HFOV (8.8 mm, not used) | {optics.datasheet_hfov_deg:.2f} deg (computed {optics.computed_2_3_hfov_deg:.3f}) |")
    a(f"| IFOV | {optics.ifov_deg:.5f} deg/px |")
    a(f"| Nadir GSD at design lat (mosaic pixel) | **{h * 1000.0 / (FOCAL_LENGTH_MM / 1000.0) * PIXEL_UM * 1e-6:.2f} m** |")
    a(f"| Nadir GSD (2×2 band cell) | {2 * h * 1000.0 / (FOCAL_LENGTH_MM / 1000.0) * PIXEL_UM * 1e-6:.2f} m |")
    a(f"| Nadir lateral half-swath | **±{r_fit:.2f} km** |")
    a(f"| 2-axis ±{AZ_BOX_DEG:.0f} deg half-swath | {r_box:.1f} km |")
    a("")
    a("## Along-track time (elevation axis, design pass)")
    a("")
    a(f"Design pass: latitude **{DESIGN_LAT_DEG:.2f} deg**, heading due east,")
    a("Earth rotation almost purely along-track.")
    a("")
    a("| Quantity | Value |")
    a("| --- | --- |")
    a(f"| Elevation window | 90 → 30 deg, one-sided, stop at limit |")
    a(f"| Time in elevation window | **{times.along_track_s:.1f} s** |")
    a(f"| Staring at nadir (no gimbal) | {times.stare_s:.2f} s |")
    a(f"| Peak elevation rate | {times.peak_el_rate_deg_s:.3f} deg/s |")
    a(f"| Peak \\|az\\| of the origin (Earth rotation) | {times.az_max_deg:.3f} deg |")
    a(f"| Window start / stop from CA | {times.t_start_s:.1f} / {times.t_stop_s:.1f} s |")
    a(f"| Slant range at start / stop | {times.slant_start_km:.1f} / {times.slant_stop_km:.1f} km |")
    a("")
    a("## Earth rotation vs pass latitude")
    a("")
    a("At low latitude the ISS heading has a large north component, so Earth")
    a("rotation (always east) has a **cross-track** piece. That walks even a")
    a("nadir-centered origin in azimuth, peaking at the 30 deg stop where range")
    a("is long. At max latitude the heading is due east and the walk vanishes.")
    a("That is why a polar-slice one-axis placement is the right one-axis case.")
    a("")
    a("| lat (deg) | h (km) | el window (s) | peak \\|az\\| origin (deg) | 1-axis R=0 (s) | 1-axis R=5 km (s) | 2-axis (s) |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    for r0, r5 in zip(lat_rows_r0, lat_rows_r5, strict=True):
        a(
            f"| {r0[0]:.2f} | {r0[1]:.1f} | {r0[2]:.1f} | {r0[3]:.3f} | "
            f"{r0[4]:.1f} | {r5[4]:.1f} | {r0[5]:.1f} |"
        )
    a("")
    a("For a point origin, Earth rotation walks it out of the chip below ~40 deg")
    a("latitude. For a 5 km cluster the threshold is ~45 deg. Below that,")
    a("one-axis only keeps the last tens of seconds near nadir. That is a")
    a("2-axis problem, or a choice not to work those passes.")
    a("")
    a("## Plant-cluster covering disk (design pass)")
    a("")
    a("Elevation tracks the cluster CoG. A plume is acquired if it is in the")
    a("frame. The covering radius is `R_cover = D + r` for stack half-span D")
    a("and per-plume CoG radius r. Equal-weight centroid tracking does *not*")
    a("grow with plume count: `R_centroid = r`. The covering disk is the")
    a("stricter (and right) metric if we want every plume in the cluster.")
    a("")
    a("Design R = 5 km: a large integrated complex (~4 km wide, e.g. Baytown")
    a("refinery) plus a few km of plume CoG. Typical small plants are 1–2 km.")
    a("")
    a("| R (km) | peak \\|az\\| (deg) | 1-axis (s) | 2-axis (s) | lost (s) | lost % |")
    a("| --- | --- | --- | --- | --- | --- |")
    for i, r in enumerate(table["radius_km"]):
        t1 = float(table["one_axis_s"][i])
        t2 = float(table["two_axis_s"][i])
        lost = float(table["lost_s"][i])
        frac = 0.0 if t2 <= 0 else 100.0 * lost / t2
        a(
            f"| {r:.1f} | {table['az_peak_deg'][i]:.2f} | {t1:.1f} | {t2:.1f} | "
            f"{lost:.1f} | {frac:.1f}% |"
        )
    a("")
    a(f"At the design pass, a disk with **R ≤ {r_fit:.1f} km** stays in the chip")
    a("for the whole elevation window. Design R = 5 km is **0 s lost**.")
    a("")
    a("Wind during the window:")
    a("")
    a("| wind (m/s) | extra displacement (km) | leftover 1-axis budget (km) |")
    a("| --- | --- | --- |")
    for w in WIND_MPS:
        extra = (w / 1000.0) * times.along_track_s
        a(f"| {w:.0f} | {extra:.2f} | {r_fit - extra:.2f} |")
    a("")
    a("## Off-center cluster (design pass, R = 0)")
    a("")
    a("| origin cross-track (km) | 1-axis (s) | 2-axis (s) | lost (s) |")
    a("| --- | --- | --- | --- |")
    for y, t1, t2 in offsets:
        a(f"| {y:.0f} | {t1:.1f} | {t2:.1f} | {t2 - t1:.1f} |")
    a("")
    a("## Full-frame inference")
    a("")
    a("Optical FOV is the full chip either way. Full-frame inference changes")
    a("compute, not tracking time. The 2×2 band plane is")
    a(f"**{BAND_LATERAL_PX} × {BAND_ALONG_PX}**.")
    a("")
    a("| | 256×256 (today) | full band plane | scale |")
    a("| --- | --- | --- | --- |")
    a(f"| Pixels | {area_256} | {area_ff} | {scale:.1f}× |")
    a(f"| DilateNet-w32 FLOPs | {SEG_FLOPS_256_G:.2f} G | {SEG_FLOPS_256_G * scale:.2f} G | {scale:.1f}× |")
    a(f"| ShuffleNet FLOPs | {CLS_FLOPS_256_G:.3f} G | {CLS_FLOPS_256_G * scale:.2f} G | {scale:.1f}× |")
    a(
        f"| CPU accept latency (linear scale from 256) | "
        f"{SEG_LAT_256_MS:.1f} / {CLS_LAT_256_MS:.1f} ms | "
        f"{SEG_LAT_256_MS * scale:.0f} / {CLS_LAT_256_MS * scale:.0f} ms | |"
    )
    a("")
    a("The 500× parameter cut (DilateNet-w32 23.5 k vs U-Net 13.4 M) is what")
    a("makes full-frame plausible. FLOPs still fit the 500 ms budget if latency")
    a("scales with pixels. Jetson Nano (4 GB) is tighter on **activation memory**")
    a("at 1224×1024 than Orin NX; that needs a bench, not a FLOP argument.")
    a("A 256-pixel centre crop on this optic is only ~0.67 deg — that would")
    a("throw away the 1-axis lateral budget. Full-frame is the right inference")
    a("size for this lens.")
    a("")
    a("## Bottom line")
    a("")
    a(f"1. Along-track, one elevation axis gives **{times.along_track_s:.0f} s**")
    a(f"   at the design (max-lat) pass, vs **{times.stare_s:.1f} s** staring.")
    a("   Stop at 30 deg. Peak rate is under 2 deg/s.")
    a(f"2. At that pass, Earth-rotation az walk of the origin is")
    a(f"   **{times.az_max_deg:.3f} deg**. A 5 km plant cluster stays in the")
    a(f"   ±{optics.half_az_deg:.2f} deg chip for the whole window — **0 s lost**")
    a("   vs two-axis.")
    a("3. At equatorial and mid-latitude passes, Earth rotation walks the")
    a("   origin out at the 30 deg stop. One-axis then keeps only the last")
    a("   ~20–60 s depending on cluster size. Use two-axis for those passes,")
    a("   or do not work them. From ~45 deg up, a 5 km cluster is lossless.")
    a("4. If two plants are farther apart than ~24 km, they are two clusters.")
    a("   Track one.")
    a("5. Worldwide stack inventory (Climate TRACE 2025) is in")
    a("   `INDUSTRIAL.md`. Stack-weighted expected 1-axis loss over the")
    a("   ISS belt is ~59% of the 121 s two-axis window, because ~90% of")
    a("   stacks sit below 45 deg latitude.")
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
    a("uv run python scratch/single-axis-gimbal-analysis/analyze.py")
    a("```")
    a("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_check(orbit: Orbit, optics: Optics, times: WindowTimes) -> None:
    u0 = argument_of_latitude(DESIGN_LAT_DEG, orbit.inclination_rad)
    r, v = iss_eci(0.0, orbit, u0)
    tgt = origin_ecef(orbit, DESIGN_LAT_DEG)
    look0 = look_at(r, v, tgt)
    assert abs(look0.el_deg - 90.0) < 0.05, look0.el_deg
    assert abs(look0.az_deg) < 0.05, look0.az_deg
    assert times.along_track_s > 90.0
    assert times.along_track_s < 160.0
    assert times.peak_el_rate_deg_s < MAX_SLEW_DEG_S
    assert times.az_max_deg < 0.05  # max-lat: Earth rotation along-track
    assert optics.fov_az_deg < optics.fov_az_raw_deg
    assert abs(optics.computed_2_3_hfov_deg - DATASHEET_HFOV_2_3_DEG) < 0.02
    # Equator must show a several-degree az walk.
    t_eq, _ = origin_window(orbit, optics, 0.0, 0.0, WINDOW_MODE)
    assert t_eq.az_max_deg > 2.0
    print("self-check: ok")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    optics = build_optics()
    orbit = build_orbit(use_perigee=False)
    orbit_p = build_orbit(use_perigee=True)
    times, data = origin_window(orbit, optics, DESIGN_LAT_DEG, 0.0, WINDOW_MODE)
    times_p, _ = origin_window(orbit_p, optics, DESIGN_LAT_DEG, 0.0, WINDOW_MODE)
    self_check(orbit, optics, times)

    table = tracking_time_vs_radius(orbit, optics, DESIGN_LAT_DEG, DISK_RADII_KM, WINDOW_MODE)
    offsets = offset_times(orbit, optics, DESIGN_LAT_DEG, WINDOW_MODE)
    lat0 = latitude_table(orbit, optics, PASS_LATS_DEG, 0.0, WINDOW_MODE)
    lat5 = latitude_table(orbit, optics, PASS_LATS_DEG, CLUSTER_R_KM, WINDOW_MODE)

    plot_along_track(orbit, optics, times, data, DESIGN_LAT_DEG, OUT / "along_track_timeline.png")
    plot_disk_angle(orbit, optics, times, DESIGN_LAT_DEG, OUT / "disk_angular_radius.png")
    plot_time_vs_radius(table, optics, times.local_alt_km, DESIGN_LAT_DEG, OUT / "tracking_time_vs_radius.png")
    plot_lost_time(table, OUT / "lost_time_vs_radius.png")
    plot_footprint(orbit, optics, times, DESIGN_LAT_DEG, OUT / "footprint_vs_time.png")
    plot_offset_map(offsets, OUT / "origin_offset.png")
    plot_required_az(table, optics, OUT / "required_az_vs_radius.png")
    plot_latitude(lat0, lat5, optics, OUT / "latitude_earth_rotation.png")

    with (OUT / "tracking_time_vs_radius.csv").open("w", encoding="utf-8") as fh:
        fh.write("radius_km,az_peak_deg,one_axis_s,two_axis_s,lost_s\n")
        for i, r in enumerate(table["radius_km"]):
            fh.write(
                f"{r:.3f},{table['az_peak_deg'][i]:.4f},"
                f"{table['one_axis_s'][i]:.3f},{table['two_axis_s'][i]:.3f},"
                f"{table['lost_s'][i]:.3f}\n"
            )
    with (OUT / "origin_offset.csv").open("w", encoding="utf-8") as fh:
        fh.write("origin_cross_km,one_axis_s,two_axis_s,lost_s\n")
        for y, t1, t2 in offsets:
            fh.write(f"{y:.1f},{t1:.3f},{t2:.3f},{t2 - t1:.3f}\n")
    with (OUT / "latitude.csv").open("w", encoding="utf-8") as fh:
        fh.write("lat_deg,h_km,el_window_s,az_max_deg,one_axis_R0_s,one_axis_R5_s,two_axis_s\n")
        for r0, r5 in zip(lat0, lat5, strict=True):
            fh.write(
                f"{r0[0]:.4f},{r0[1]:.3f},{r0[2]:.3f},{r0[3]:.4f},{r0[4]:.3f},{r5[4]:.3f},{r0[5]:.3f}\n"
            )

    write_report(
        orbit,
        optics,
        times,
        table,
        offsets,
        lat0,
        lat5,
        times_p,
        ROOT / "RESULTS.md",
    )

    print()
    print(f"TLE epoch         {TLE_EPOCH}")
    print(f"design lat        {DESIGN_LAT_DEG:.2f} deg")
    print(f"local altitude    {times.local_alt_km:.2f} km")
    print(f"usable FOV        {optics.fov_az_deg:.3f} x {optics.fov_el_deg:.3f} deg")
    print(f"along-track time  {times.along_track_s:.1f} s")
    print(f"origin |az| max   {times.az_max_deg:.3f} deg")
    print(f"stare             {times.stare_s:.2f} s")
    print(f"peak el rate      {times.peak_el_rate_deg_s:.3f} deg/s")
    print(f"nadir half-swath  {times.local_alt_km * math.tan(math.radians(optics.half_az_deg)):.2f} km")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
