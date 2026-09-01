#!/usr/bin/env python3
"""TEMPORARY ANALYSIS — not flight software.

Single-axis vs two-axis gimbal tracking time for an ISS polar-slice view.

This folder is a working analysis, not a package. Edit the ASSUMPTIONS block
and re-run:

    uv run python scratch/single-axis-gimbal-analysis/analyze.py

Outputs land in scratch/single-axis-gimbal-analysis/outputs/.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# ASSUMPTIONS — edit these, then re-run.
# ---------------------------------------------------------------------------

# ISS circular-orbit baseline. Sensitivity also runs 400 km and 420 km.
ALTITUDE_KM = 400.0
EARTH_RADIUS_KM = 6371.0  # mean spherical Earth
MU_KM3_S2 = 398600.4418  # Earth gravitational parameter

# Operational pointing box. Elevation 90 deg = mount normal (nadir if untilted).
# Elevation 30 deg = 60 deg off-nadir in the ISS-velocity direction.
AZ_BOX_DEG = 10.0
EL_NADIR_DEG = 90.0
EL_LIMB_DEG = 30.0  # 90 - 30 = 60 deg along-track off-nadir

# Hardware: FLIR BFS-U3-50S5 (Sony IMX264) + 150 mm athermal C-mount.
PIXEL_UM = 3.45
N_LATERAL_PX = 2448  # long axis: cross-track / azimuth
N_ALONG_PX = 2048  # short axis: along-track / elevation
FOCAL_LENGTH_MM = 150.0
# Lens datasheet HFOV for a *standard* 8.8 mm 2/3" format (not the IMX264 width).
DATASHEET_2_3_WIDTH_MM = 8.8
DATASHEET_HFOV_2_3_DEG = 3.36
LENS_DISTORTION_PCT = 0.66

# Flight-software slew cap (mission envelope, not hardware).
MAX_SLEW_DEG_S = 2.0

# CoG-disk radii to sweep (km on the ground).
DISK_RADII_KM = (0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0)

# Wind advection during a pass (m/s). Extra ground displacement = v * T_track.
WIND_MPS = (0.0, 10.0, 20.0)

# Origin cross-track offsets (km) for "target anywhere in the operational box".
ORIGIN_OFFSETS_KM = (0.0, 5.0, 10.0, 20.0, 40.0, 70.0)

# One-sided window: only the approach (or only the recede), el in [30, 90].
# Two-sided: a single elevation axis that travels through nadir to the other 30 deg.
WINDOW_MODE = "one_sided"

DT_S = 0.05

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

# Colorblind-friendly.
C_BLUE = "#0077BB"
C_ORANGE = "#EE7733"
C_TEAL = "#009988"
C_RED = "#CC3311"
C_SKY = "#33BBEE"
C_INK = "#222222"


# ---------------------------------------------------------------------------
# Optics
# ---------------------------------------------------------------------------


def fov_deg(n_px: int, pixel_um: float, fl_mm: float) -> float:
    """Full-width FOV from thin-lens geometry: 2 atan((N p / 2) / f)."""
    half_mm = (n_px * pixel_um / 1000.0) / 2.0
    return 2.0 * math.degrees(math.atan(half_mm / fl_mm))


def ifov_deg_per_px(pixel_um: float, fl_mm: float) -> float:
    return math.degrees(math.atan((pixel_um / 1000.0) / fl_mm))


@dataclass(frozen=True)
class Optics:
    fov_az_deg: float
    fov_el_deg: float
    ifov_deg: float
    gsd_nadir_m: float
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


def build_optics(altitude_km: float) -> Optics:
    fov_az = fov_deg(N_LATERAL_PX, PIXEL_UM, FOCAL_LENGTH_MM)
    fov_el = fov_deg(N_ALONG_PX, PIXEL_UM, FOCAL_LENGTH_MM)
    ifov = ifov_deg_per_px(PIXEL_UM, FOCAL_LENGTH_MM)
    width_mm = N_LATERAL_PX * PIXEL_UM / 1000.0
    height_mm = N_ALONG_PX * PIXEL_UM / 1000.0
    gsd = (altitude_km * 1000.0) / (FOCAL_LENGTH_MM / 1000.0) * (PIXEL_UM * 1e-6)
    computed_23 = fov_deg(
        int(round(DATASHEET_2_3_WIDTH_MM / (PIXEL_UM / 1000.0))),
        PIXEL_UM,
        FOCAL_LENGTH_MM,
    )
    # Direct datasheet-format FOV from 8.8 mm width, independent of pixel count.
    computed_23 = 2.0 * math.degrees(math.atan((DATASHEET_2_3_WIDTH_MM / 2.0) / FOCAL_LENGTH_MM))
    return Optics(
        fov_az_deg=fov_az,
        fov_el_deg=fov_el,
        ifov_deg=ifov,
        gsd_nadir_m=gsd,
        sensor_width_mm=width_mm,
        sensor_height_mm=height_mm,
        datasheet_hfov_deg=DATASHEET_HFOV_2_3_DEG,
        computed_2_3_hfov_deg=computed_23,
    )


# ---------------------------------------------------------------------------
# Orbit and look geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Orbit:
    altitude_km: float
    radius_km: float
    n_rad_s: float
    v_km_s: float
    period_s: float
    eta_horizon_deg: float

    @property
    def ground_track_km_s(self) -> float:
        return self.v_km_s * EARTH_RADIUS_KM / self.radius_km


def build_orbit(altitude_km: float) -> Orbit:
    a = EARTH_RADIUS_KM + altitude_km
    n = math.sqrt(MU_KM3_S2 / a**3)
    v = math.sqrt(MU_KM3_S2 / a)
    eta_h = math.degrees(math.asin(EARTH_RADIUS_KM / a))
    return Orbit(
        altitude_km=altitude_km,
        radius_km=a,
        n_rad_s=n,
        v_km_s=v,
        period_s=2.0 * math.pi / n,
        eta_horizon_deg=eta_h,
    )


def iss_state(t_s: float, orbit: Orbit) -> tuple[np.ndarray, np.ndarray]:
    """ISS position and inertial velocity. t=0 is closest approach to the origin.

    ECEF-like working frame (spherical, non-rotating):
      x = along-track, y = cross-track, z = geocentric radial at CA.
    """
    delta = orbit.n_rad_s * t_s
    s = orbit.radius_km * np.array([math.sin(delta), 0.0, math.cos(delta)])
    v = orbit.radius_km * orbit.n_rad_s * np.array([math.cos(delta), 0.0, -math.sin(delta)])
    return s, v


def ground_point(along_km: float, cross_km: float) -> np.ndarray:
    """Earth-surface point near the CA origin, by local tangent-plane offset."""
    origin = np.array([0.0, 0.0, EARTH_RADIUS_KM])
    east = np.array([1.0, 0.0, 0.0])  # along-track at CA
    north = np.array([0.0, 1.0, 0.0])  # cross-track
    vec = origin + east * along_km + north * cross_km
    return vec / np.linalg.norm(vec) * EARTH_RADIUS_KM


def body_axes(s: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x along-track, y cross-track, z nadir (toward Earth)."""
    z = -s / np.linalg.norm(s)
    x = v - np.dot(v, z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    return x, y, z


@dataclass(frozen=True)
class Look:
    az_deg: float  # cross-track gimbal angle; 0 in the along-track / nadir plane
    el_deg: float  # 90 at nadir, 30 at 60 deg forward off-nadir, >90 if looking aft
    eta_deg: float  # total off-nadir
    slant_km: float
    visible: bool
    along_off_nadir_deg: float
    cross_off_nadir_deg: float


def look_at(s: np.ndarray, v: np.ndarray, target: np.ndarray) -> Look:
    x, y, z = body_axes(s, v)
    look = target - s
    slant = float(np.linalg.norm(look))
    if slant < 1e-9:
        return Look(0.0, 90.0, 0.0, 0.0, False, 0.0, 0.0)
    u = look / slant
    lx = float(np.dot(look, x))
    ly = float(np.dot(look, y))
    lz = float(np.dot(look, z))  # nadir component of the look vector
    eta = math.degrees(math.acos(float(np.clip(np.dot(u, z), -1.0, 1.0))))
    # Tip-tilt box: elevation from the along-track/nadir projection, azimuth
    # from the cross-track / nadir projection. Both are defined at nadir.
    along_off = math.degrees(math.atan2(lx, lz))
    cross_off = math.degrees(math.atan2(ly, lz))
    el = EL_NADIR_DEG - along_off
    az = math.degrees(math.atan2(ly, math.hypot(lx, lz)))
    visible = _hits_earth(s, u)
    return Look(
        az_deg=az,
        el_deg=el,
        eta_deg=eta,
        slant_km=slant,
        visible=visible,
        along_off_nadir_deg=along_off,
        cross_off_nadir_deg=cross_off,
    )


def _hits_earth(s: np.ndarray, u: np.ndarray) -> bool:
    """Ray from ISS along unit u intersects the spherical Earth in front of the camera."""
    b = 2.0 * float(np.dot(s, u))
    c = float(np.dot(s, s)) - EARTH_RADIUS_KM**2
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return False
    sqrt_d = math.sqrt(disc)
    t1 = (-b - sqrt_d) / 2.0
    t2 = (-b + sqrt_d) / 2.0
    return t1 > 0.0 or t2 > 0.0


def earth_hit(s: np.ndarray, u: np.ndarray) -> np.ndarray | None:
    b = 2.0 * float(np.dot(s, u))
    c = float(np.dot(s, s)) - EARTH_RADIUS_KM**2
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None
    sqrt_d = math.sqrt(disc)
    t1 = (-b - sqrt_d) / 2.0
    t2 = (-b + sqrt_d) / 2.0
    ts = [t for t in (t1, t2) if t > 1e-6]
    if not ts:
        return None
    return s + min(ts) * u


# ---------------------------------------------------------------------------
# Sampling a pass
# ---------------------------------------------------------------------------


def sample_pass(
    orbit: Orbit,
    along_km: float,
    cross_km: float,
    t_min: float,
    t_max: float,
    dt: float = DT_S,
) -> dict[str, np.ndarray]:
    target = ground_point(along_km, cross_km)
    ts = np.arange(t_min, t_max + dt * 0.5, dt)
    n = ts.size
    az = np.empty(n)
    el = np.empty(n)
    eta = np.empty(n)
    slant = np.empty(n)
    along = np.empty(n)
    cross = np.empty(n)
    vis = np.empty(n, dtype=bool)
    for i, t in enumerate(ts):
        s, v = iss_state(float(t), orbit)
        look = look_at(s, v, target)
        az[i] = look.az_deg
        el[i] = look.el_deg
        eta[i] = look.eta_deg
        slant[i] = look.slant_km
        along[i] = look.along_off_nadir_deg
        cross[i] = look.cross_off_nadir_deg
        vis[i] = look.visible
    return {
        "t": ts,
        "az": az,
        "el": el,
        "eta": eta,
        "slant": slant,
        "along": along,
        "cross": cross,
        "vis": vis,
    }


def in_elevation_window(el_deg: np.ndarray, mode: str = WINDOW_MODE) -> np.ndarray:
    """True when the along-track gimbal can point at this elevation.

    one_sided: el in [30, 90]  (forward look only; aft maps to el > 90).
    two_sided: the same 60 deg of off-nadir on both sides of nadir.
    """
    forward = (el_deg >= EL_LIMB_DEG) & (el_deg <= EL_NADIR_DEG)
    if mode == "one_sided":
        return forward
    # Aft looks have el = 90 - (negative along-off) = 90 + |along_off|.
    # Two-sided: |90 - el| <= 60, i.e. el in [30, 150], and still Earth-visible.
    return np.abs(el_deg - EL_NADIR_DEG) <= (EL_NADIR_DEG - EL_LIMB_DEG)


def mask_time_s(mask: np.ndarray, t: np.ndarray) -> float:
    """Integrated time the mask is true (does not fill gaps)."""
    if t.size < 2 or not np.any(mask):
        return 0.0
    dt = float(np.median(np.diff(t)))
    return float(np.sum(mask) * dt)


def angular_rate_deg_s(t: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    rate = np.gradient(angle_deg, t)
    return rate


# ---------------------------------------------------------------------------
# In-frame tests
# ---------------------------------------------------------------------------


def in_sensor_frame(
    az_err_deg: np.ndarray,
    el_err_deg: np.ndarray,
    optics: Optics,
) -> np.ndarray:
    return (np.abs(az_err_deg) <= optics.half_az_deg) & (np.abs(el_err_deg) <= optics.half_el_deg)


def two_axis_boresightable(az: np.ndarray, el: np.ndarray, mode: str) -> np.ndarray:
    return in_elevation_window(el, mode) & (np.abs(az) <= AZ_BOX_DEG)


def one_axis_in_frame(
    az: np.ndarray,
    el: np.ndarray,
    optics: Optics,
    mode: str,
) -> np.ndarray:
    """Elevation axis tracks the origin; azimuth is parked at 0.

    Inside the elevation window the along-track error is zero. After the
    forward stop the gimbal holds 30 deg and the target walks out through the
    leftover along-track FOV. Aft of nadir is not tracked in one-sided mode.
    """
    in_el = in_elevation_window(el, mode)
    az_ok = np.abs(az) <= optics.half_az_deg
    residual_forward = np.abs(el - EL_LIMB_DEG) <= optics.half_el_deg
    if mode == "two_sided":
        residual_aft = np.abs(el - (EL_NADIR_DEG + (EL_NADIR_DEG - EL_LIMB_DEG))) <= optics.half_el_deg
        el_ok = in_el | residual_forward | residual_aft
    else:
        el_ok = in_el | residual_forward
    return az_ok & el_ok


def staring_in_frame(az: np.ndarray, el: np.ndarray, optics: Optics) -> np.ndarray:
    """Gimbal parked at nadir (el=90, az=0)."""
    return in_sensor_frame(az, el - EL_NADIR_DEG, optics)


def disk_points(radius_km: float, n: int = 72) -> np.ndarray:
    phis = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    along = radius_km * np.cos(phis)
    cross = radius_km * np.sin(phis)
    pts = np.stack([along, cross], axis=1)
    return pts


def disk_max_az(
    orbit: Orbit,
    t: float,
    origin_cross_km: float,
    radius_km: float,
) -> float:
    """Worst-case |az| over a ground disk around (0, origin_cross)."""
    s, v = iss_state(t, orbit)
    worst = 0.0
    for along, cross in disk_points(radius_km):
        tgt = ground_point(float(along), origin_cross_km + float(cross))
        look = look_at(s, v, tgt)
        worst = max(worst, abs(look.az_deg))
    # Include the origin itself.
    look0 = look_at(s, v, ground_point(0.0, origin_cross_km))
    worst = max(worst, abs(look0.az_deg))
    return worst


# ---------------------------------------------------------------------------
# Derived timelines
# ---------------------------------------------------------------------------


@dataclass
class WindowTimes:
    along_track_s: float
    residual_s: float
    total_in_frame_s: float
    stare_s: float
    peak_el_rate_deg_s: float
    t_start_s: float
    t_stop_s: float
    slant_start_km: float
    slant_stop_km: float


def origin_window(
    orbit: Orbit,
    optics: Optics,
    origin_cross_km: float,
    mode: str,
) -> tuple[WindowTimes, dict[str, np.ndarray]]:
    # Span enough true-anomaly that 60 deg off-nadir is inside the sample.
    t_guess = 250.0
    data = sample_pass(orbit, 0.0, origin_cross_km, -t_guess, t_guess)
    el_mask = in_elevation_window(data["el"], mode) & data["vis"]
    if not np.any(el_mask):
        empty = WindowTimes(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return empty, data
    t = data["t"]
    idx = np.where(el_mask)[0]
    t_start = float(t[idx[0]])
    t_stop = float(t[idx[-1]])
    along_s = t_stop - t_start
    one = one_axis_in_frame(data["az"], data["el"], optics, mode)
    in_frame_s = mask_time_s(one, t)
    residual = max(0.0, in_frame_s - along_s)
    stare = mask_time_s(staring_in_frame(data["az"], data["el"], optics), t)
    rate = np.abs(angular_rate_deg_s(t, data["el"]))
    peak_rate = float(np.max(rate[el_mask])) if np.any(el_mask) else 0.0
    slant_start = float(data["slant"][idx[0]])
    slant_stop = float(data["slant"][idx[-1]])
    times = WindowTimes(
        along_track_s=along_s,
        residual_s=residual,
        total_in_frame_s=in_frame_s,
        stare_s=stare,
        peak_el_rate_deg_s=peak_rate,
        t_start_s=t_start,
        t_stop_s=t_stop,
        slant_start_km=slant_start,
        slant_stop_km=slant_stop,
    )
    return times, data


def tracking_time_vs_radius(
    orbit: Orbit,
    optics: Optics,
    origin_cross_km: float,
    radii: tuple[float, ...],
    mode: str,
) -> dict[str, np.ndarray]:
    """In-frame time for a CoG that can sit anywhere on the disk (worst-case edge).

    Two-axis: gimbal tracks the CoG as long as the required (az, el) is in the
    operational box. One-axis: elevation tracks the *origin*; the CoG stays in
    frame only while |az_cog| <= sensor half-FOV and elevation is in (or just
    past) the window.
    """
    t_guess = 250.0
    t_grid = np.arange(-t_guess, t_guess + DT_S, DT_S)
    r_arr = np.array(radii, dtype=float)
    t1 = np.zeros(r_arr.size)
    t2 = np.zeros(r_arr.size)
    az_peak = np.zeros(r_arr.size)
    for i, r in enumerate(r_arr):
        # Worst-case CoG is the cross-track edge (along-track edge is followed
        # by the elevation axis). Sample the +cross edge; the disk is symmetric.
        edge = sample_pass(orbit, 0.0, origin_cross_km + r, -t_guess, t_guess, DT_S)
        origin = sample_pass(orbit, 0.0, origin_cross_km, -t_guess, t_guess, DT_S)
        # Compare inside the origin's elevation window only (operational dwell).
        # One-axis parks az at 0 and follows the origin in elevation, so the CoG
        # stays in frame while |az_edge| fits in the sensor. Two-axis follows
        # the CoG until it leaves the operational box.
        el_ok = in_elevation_window(origin["el"], mode) & origin["vis"]
        one = el_ok & (np.abs(edge["az"]) <= optics.half_az_deg)
        two = two_axis_boresightable(edge["az"], edge["el"], mode)
        t1[i] = mask_time_s(one, edge["t"])
        t2[i] = mask_time_s(two, edge["t"])
        az_sel = np.abs(edge["az"][el_ok])
        az_peak[i] = float(np.max(az_sel)) if az_sel.size else 0.0
    return {
        "radius_km": r_arr,
        "one_axis_s": t1,
        "two_axis_s": t2,
        "lost_s": t2 - t1,
        "az_peak_deg": az_peak,
        "t_grid": t_grid,
    }


def footprint_half_width_km(
    orbit: Orbit,
    optics: Optics,
    t: float,
    origin_cross_km: float,
) -> tuple[float, float]:
    """Ground half-width of the sensor FOV at the origin, along and across track."""
    s, v = iss_state(t, orbit)
    x, y, z = body_axes(s, v)
    origin = ground_point(0.0, origin_cross_km)
    look0 = origin - s
    u0 = look0 / np.linalg.norm(look0)
    # Rotate the boresight by ±half-FOV about body y (along-track FOV) and
    # about body x (cross-track FOV).
    half_el = math.radians(optics.half_el_deg)
    half_az = math.radians(optics.half_az_deg)

    def rot(vec: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
        axis = axis / np.linalg.norm(axis)
        k = axis
        return (
            vec * math.cos(ang)
            + np.cross(k, vec) * math.sin(ang)
            + k * np.dot(k, vec) * (1.0 - math.cos(ang))
        )

    u_el = rot(u0, y, half_el)
    u_az_pos = rot(u0, x, half_az)
    p0 = earth_hit(s, u0)
    p_el = earth_hit(s, u_el)
    p_az = earth_hit(s, u_az_pos)
    if p0 is None or p_el is None or p_az is None:
        return float("nan"), float("nan")
    along_km = float(np.linalg.norm(p_el - p0))
    cross_km = float(np.linalg.norm(p_az - p0))
    return along_km, cross_km


# ---------------------------------------------------------------------------
# Plots and report
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


def plot_geometry(orbit: Orbit, optics: Optics, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    re, a = EARTH_RADIUS_KM, orbit.radius_km
    earth = plt.Circle((0, 0), re, fill=True, color="#cfe8d8", lw=0)
    ax.add_patch(earth)
    globe = plt.Circle((0, 0), re, fill=False, color="#335544", lw=1.2)
    ax.add_patch(globe)
    # ISS at CA.
    ax.plot(0, a, "o", color=C_INK, ms=7, zorder=5)
    ax.annotate("ISS", (0, a), xytext=(18, 8), textcoords="offset points")
    # Nadir and 60 deg look.
    ax.plot([0, 0], [a, re], color=C_BLUE, lw=1.8, label="el = 90 deg (nadir)")
    times, _data = origin_window(orbit, optics, 0.0, "one_sided")
    s_stop, _ = iss_state(times.t_start_s, orbit)
    tgt_origin = ground_point(0.0, 0.0)
    ax.plot(
        [s_stop[0], tgt_origin[0]],
        [s_stop[2], tgt_origin[2]],
        color=C_ORANGE,
        lw=1.8,
        label=f"el = {EL_LIMB_DEG:.0f} deg (window edge)",
    )
    ax.plot(s_stop[0], s_stop[2], "o", color=C_ORANGE, ms=6)
    # Sensor FOV at nadir (tiny cone).
    half = math.radians(optics.half_el_deg)
    for sign in (-1.0, 1.0):
        u = np.array([math.sin(sign * half), 0.0, -math.cos(sign * half)])
        hit = earth_hit(np.array([0.0, 0.0, a]), u)
        if hit is not None:
            ax.plot([0, hit[0]], [a, hit[2]], color=C_TEAL, lw=1.0, ls="--")
    ax.plot([], [], color=C_TEAL, ls="--", label=f"sensor el FOV {optics.fov_el_deg:.2f} deg")
    # 10 km along-track bar at the origin — the CoG disk is mostly out of plane.
    r = 10.0
    ax.plot([-r, r], [re, re], color=C_RED, lw=3, solid_capstyle="round", label="10 km along-track at origin")
    ax.set_aspect("equal")
    x_edge = abs(float(s_stop[0])) + 250.0
    ax.set_xlim(-x_edge, x_edge)
    ax.set_ylim(re - 40, a + 80)
    ax.set_xlabel("along-track ECEF x (km)")
    ax.set_ylabel("geocentric z (km)")
    ax.set_title("Elevation window vs sensor FOV (along-track plane)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_along_track(orbit: Orbit, optics: Optics, times: WindowTimes, data: dict, path: Path) -> None:
    t = data["t"]
    mask = (t >= times.t_start_s - 15) & (t <= times.t_stop_s + 15)
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.2), sharex=True)
    axes[0].plot(t[mask], data["el"][mask], color=C_BLUE, lw=2)
    axes[0].axhline(EL_NADIR_DEG, color=C_INK, ls=":", lw=1)
    axes[0].axhline(EL_LIMB_DEG, color=C_ORANGE, ls="--", lw=1, label="el stop 30 deg")
    axes[0].axvspan(times.t_start_s, times.t_stop_s, color=C_BLUE, alpha=0.12, label="elevation window")
    axes[0].set_ylabel("elevation (deg)")
    axes[0].set_ylim(20, 100)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title(
        f"Along-track track time = {times.along_track_s:.1f} s "
        f"(h = {orbit.altitude_km:.0f} km, one-sided {EL_NADIR_DEG:.0f}→{EL_LIMB_DEG:.0f} deg)"
    )

    rate = np.abs(angular_rate_deg_s(t, data["el"]))
    axes[1].plot(t[mask], rate[mask], color=C_ORANGE, lw=2)
    axes[1].axhline(MAX_SLEW_DEG_S, color=C_RED, ls="--", label=f"mission slew cap {MAX_SLEW_DEG_S:.0f} deg/s")
    axes[1].set_ylabel("|d(el)/dt| (deg/s)")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(t[mask], data["slant"][mask], color=C_TEAL, lw=2)
    axes[2].set_ylabel("slant range (km)")
    axes[2].set_xlabel("time from closest approach (s)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_disk_angle(
    orbit: Orbit,
    optics: Optics,
    times: WindowTimes,
    path: Path,
) -> None:
    t = np.arange(times.t_start_s, times.t_stop_s + DT_S, 1.0)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    radii = (1.0, 2.0, 5.0, 10.0, 15.0, 20.0)
    colors = [C_SKY, C_BLUE, C_TEAL, C_ORANGE, C_RED, "#882255"]
    for r, c in zip(radii, colors, strict=True):
        az = [disk_max_az(orbit, float(ti), 0.0, r) for ti in t]
        ax.plot(t, az, color=c, lw=2, label=f"R = {r:.0f} km")
    ax.axhline(
        optics.half_az_deg,
        color=C_INK,
        ls="--",
        lw=1.4,
        label=f"sensor half-FOV az = {optics.half_az_deg:.2f} deg",
    )
    ax.axhline(AZ_BOX_DEG, color=C_INK, ls=":", lw=1.4, label=f"2-axis az box ±{AZ_BOX_DEG:.0f} deg")
    ax.set_xlabel("time from closest approach (s)")
    ax.set_ylabel("worst-case |az| of CoG disk (deg)")
    ax.set_title("Lateral angular size of a ground disk while elevation-tracking the origin")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_time_vs_radius(table: dict[str, np.ndarray], optics: Optics, path: Path) -> None:
    r = table["radius_km"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(r, table["two_axis_s"], color=C_BLUE, lw=2.4, marker="o", label="two-axis (box ±10 deg az)")
    ax.plot(r, table["one_axis_s"], color=C_ORANGE, lw=2.4, marker="s", label="one-axis (az fixed at 0)")
    ax.set_xlabel("CoG disk radius R (km)")
    ax.set_ylabel("in-frame tracking time (s)")
    ax.set_title("Tracking time vs CoG-disk radius (origin on the slice centerline)")
    ax.axvline(
        ALTITUDE_KM * math.tan(math.radians(optics.half_az_deg)),
        color=C_TEAL,
        ls="--",
        label="R = h tan(FOV_az / 2) at nadir",
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_lost_time(table: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(table["radius_km"], table["lost_s"], color=C_RED, lw=2.4, marker="o")
    ax.set_xlabel("CoG disk radius R (km)")
    ax.set_ylabel("tracking time lost vs two-axis (s)")
    ax.set_title("Time given up by dropping the azimuth axis")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_footprint(orbit: Orbit, optics: Optics, times: WindowTimes, path: Path) -> None:
    t = np.linspace(times.t_start_s, times.t_stop_s, 80)
    along = []
    cross = []
    for ti in t:
        a, c = footprint_half_width_km(orbit, optics, float(ti), 0.0)
        along.append(a)
        cross.append(c)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(t, along, color=C_BLUE, lw=2, label="along-track half-footprint")
    ax.plot(t, cross, color=C_ORANGE, lw=2, label="cross-track half-footprint")
    ax.set_xlabel("time from closest approach (s)")
    ax.set_ylabel("ground half-width of sensor FOV (km)")
    ax.set_title("Instantaneous sensor footprint on the ground (elevation-tracked origin)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def offset_times(
    orbit: Orbit,
    optics: Optics,
    mode: str,
) -> list[tuple[float, float, float]]:
    """(cross_km, one_axis_s, two_axis_s) inside the elevation window."""
    rows: list[tuple[float, float, float]] = []
    for y in ORIGIN_OFFSETS_KM:
        _times, data = origin_window(orbit, optics, y, mode)
        el_ok = in_elevation_window(data["el"], mode) & data["vis"]
        one = el_ok & (np.abs(data["az"]) <= optics.half_az_deg)
        two = two_axis_boresightable(data["az"], data["el"], mode)
        rows.append((float(y), mask_time_s(one, data["t"]), mask_time_s(two, data["t"])))
    return rows


def plot_offset_map(rows: list[tuple[float, float, float]], path: Path) -> None:
    """For each origin offset, 1-axis vs 2-axis time (R=0, a single stack)."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ys = [r[0] for r in rows]
    t1 = [r[1] for r in rows]
    t2 = [r[2] for r in rows]
    ax.plot(ys, t2, color=C_BLUE, lw=2.4, marker="o", label="two-axis")
    ax.plot(ys, t1, color=C_ORANGE, lw=2.4, marker="s", label="one-axis")
    ax.set_xlabel("stack origin cross-track offset (km)")
    ax.set_ylabel("in-frame tracking time (s)")
    ax.set_title("If the stack is not on the slice centerline (R = 0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_required_az(table: dict[str, np.ndarray], optics: Optics, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(table["radius_km"], table["az_peak_deg"], color=C_BLUE, lw=2.4, marker="o")
    ax.axhline(optics.half_az_deg, color=C_ORANGE, ls="--", label="1-axis: sensor half-FOV")
    ax.axhline(AZ_BOX_DEG, color=C_INK, ls=":", label="2-axis operational az box")
    ax.set_xlabel("CoG disk radius R (km)")
    ax.set_ylabel("peak |az| during the elevation window (deg)")
    ax.set_title("Azimuth travel a 2-axis gimbal would use to keep the disk edge on boresight")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(
    orbit: Orbit,
    optics: Optics,
    times: WindowTimes,
    table: dict[str, np.ndarray],
    offsets: list[tuple[float, float, float]],
    orbit_420: Orbit | None,
    times_420: WindowTimes | None,
    times_two_sided: WindowTimes | None,
    path: Path,
) -> None:
    r_fit = orbit.altitude_km * math.tan(math.radians(optics.half_az_deg))
    r_box = orbit.altitude_km * math.tan(math.radians(AZ_BOX_DEG))
    # Wind extra radius during the along-track window.
    wind_rows = []
    for w in WIND_MPS:
        extra_km = (w / 1000.0) * times.along_track_s
        wind_rows.append((w, extra_km, r_fit - extra_km))

    # Superposition examples.
    # Centroid of N equal-weight plumes each of radius r: R_c = r (not N r).
    # Coverage of all plumes with origins inside D of a centre: R_cover = D + r.

    lines: list[str] = []
    a = lines.append
    a("# Single-axis gimbal tracking time")
    a("")
    a("TEMPORARY ANALYSIS. Generated by `analyze.py`. Not flight software.")
    a("")
    a("## Questions that still change the number")
    a("")
    a("These are the knobs that move the result by more than a few percent.")
    a("Defaults used for this run are in parentheses.")
    a("")
    a("1. **Window sidedness.** Is elevation `[90, 30]` one side of nadir only")
    a("   (~one pass of ~100 s), or can a single elevation axis travel *through*")
    a("   nadir to 30 deg on the other side (about 2× the time)? (default: one-sided)")
    a("2. **Mount tilt.** Is the mount normal geocentric nadir, or is the payload")
    a("   already tilted toward the pole? A fixed off-nadir bias changes both")
    a("   dwell and GSD. (default: untilted nadir at el = 90)")
    a("3. **CoG disk radius R.** What radius should we design to? A single visible")
    a("   industrial plume is typically a few hundred metres to a couple of km;")
    a("   a plant cluster is a few km; a city-scale set of stacks can be 10–30 km.")
    a("   **This is the number that decides whether 1-axis loses any time.**")
    a("4. **What is the tracked target?** Combined centroid of every plume")
    a("   (`R_centroid = Σ w_i r_i`, so equal-radius plumes do *not* grow with N),")
    a("   the currently selected plume (then origin *separation* matters), or the")
    a("   bounding circle that keeps every plume in the frame (`R_cover = D + r`)?")
    a("5. **In-frame vs on-boresight.** Is a plume useful anywhere in the sensor")
    a("   FOV, or must the gimbal keep the CoG on boresight (the current TRACKING")
    a("   ROI crop assumes we can centre the target)?")
    a("6. **Altitude.** 400 km baseline, or the current ISS band ~410–420 km?")
    a("7. **After the elevation stop.** Count the extra seconds while the target")
    a("   walks out of the leftover along-track FOV, or stop at the gimbal limit?")
    a("8. **FOV source.** IMX264 active area (this run) or the lens-sheet 2/3\"")
    a("   format HFOV of 3.36 deg (8.8 mm width, slightly larger than IMX264)?")
    a("9. **Full chip vs inference ROI.** Full 2448×2048 mosaic FOV, or the 256×256")
    a("   inference window the flight software uses today?")
    a("10. **Earth rotation / heading.** First-order spherical non-rotating Earth")
    a("    (this run), or a polar-latitude pass with Earth rotation and ISS")
    a("    inclination 51.6 deg?")
    a("")
    a("## Assumptions used in this run")
    a("")
    a(f"- Spherical Earth, radius {EARTH_RADIUS_KM:.0f} km, no Earth rotation.")
    a(f"- Circular ISS orbit, altitude **{orbit.altitude_km:.0f} km**.")
    a(f"- Mount normal = geocentric nadir at closest approach. Elevation window")
    a(f"  **{EL_NADIR_DEG:.0f} → {EL_LIMB_DEG:.0f} deg** ({EL_NADIR_DEG - EL_LIMB_DEG:.0f} deg off-nadir),")
    a(f"  **{WINDOW_MODE}**.")
    a(f"- Azimuth operational box ±{AZ_BOX_DEG:.0f} deg (two-axis only).")
    a("- Sensor long axis (2448 px) is **lateral** (cross-track). Short axis")
    a("  (2048 px) is along-track.")
    a("- Thin-lens FOV from IMX264 active area and 150.00 mm focal length.")
    a("  Distortion (0.66 %) is noted, not modelled.")
    a("- A ground-fixed stack origin. The tracked CoG stays in a disk of radius R")
    a("  around that origin. Worst-case CoG is the cross-track edge of the disk.")
    a("- One-axis: elevation tracks the origin; azimuth is parked at 0.")
    a("- Two-axis: both axes track the CoG while it stays in the operational box.")
    a("")
    a("## Optics (BFS-U3-50S5 + 150 mm)")
    a("")
    a("| Quantity | Value |")
    a("| --- | --- |")
    a(f"| Pixel pitch | {PIXEL_UM} µm |")
    a(f"| Array | {N_LATERAL_PX} × {N_ALONG_PX} (lateral × along-track) |")
    a(f"| Active area | {optics.sensor_width_mm:.3f} × {optics.sensor_height_mm:.3f} mm |")
    a(f"| Focal length | {FOCAL_LENGTH_MM:.2f} mm |")
    a(f"| IFOV | {optics.ifov_deg:.5f} deg/px ({optics.ifov_deg * 3600:.2f} arcsec/px) |")
    a(f"| Sensor FOV (this camera) | **{optics.fov_az_deg:.3f} deg az × {optics.fov_el_deg:.3f} deg el** |")
    a(f"| Half-FOV | ±{optics.half_az_deg:.3f} deg az, ±{optics.half_el_deg:.3f} deg el |")
    a(f"| Lens sheet HFOV for 8.8 mm 2/3\" | {optics.datasheet_hfov_deg:.2f} deg (computed {optics.computed_2_3_hfov_deg:.3f} deg) |")
    a(f"| Nadir GSD (mosaic pixel) | **{optics.gsd_nadir_m:.2f} m** |")
    a(f"| Nadir GSD (2×2 band cell) | {2 * optics.gsd_nadir_m:.2f} m |")
    a(f"| Nadir lateral half-swath | **{r_fit:.2f} km** |")
    a(f"| Nadir 2-axis ±{AZ_BOX_DEG:.0f} deg half-swath | {r_box:.1f} km |")
    a(f"| Max distortion (sheet) | {LENS_DISTORTION_PCT:.2f} % (~{LENS_DISTORTION_PCT / 100 * optics.fov_az_deg:.3f} deg FOV) |")
    a("")
    a("The flight-software placeholder `ifov_deg_per_px = 0.02` (512 band-plane")
    a(f"px → 10.24 deg) is about **{0.02 / optics.ifov_deg:.0f}×** coarser than this")
    a("optic. SIL pointing tests do not represent this lens.")
    a("")
    a("## Orbit")
    a("")
    a("| Quantity | Value |")
    a("| --- | --- |")
    a(f"| Altitude | {orbit.altitude_km:.1f} km |")
    a(f"| Orbital radius | {orbit.radius_km:.1f} km |")
    a(f"| Orbital speed | {orbit.v_km_s:.3f} km/s |")
    a(f"| Period | {orbit.period_s / 60.0:.2f} min |")
    a(f"| Mean motion | {math.degrees(orbit.n_rad_s):.4f} deg/s |")
    a(f"| Ground-track speed (no rotation) | {orbit.ground_track_km_s:.3f} km/s |")
    a(f"| Horizon off-nadir | {orbit.eta_horizon_deg:.2f} deg |")
    a("")
    a("## Along-track tracking time (the elevation axis)")
    a("")
    a("For a stack on the slice centerline the azimuth of the origin stays ~0.")
    a("The elevation axis is the one that buys tracking time.")
    a("")
    a("| Quantity | Value |")
    a("| --- | --- |")
    a(f"| Elevation window | {EL_NADIR_DEG:.0f} → {EL_LIMB_DEG:.0f} deg ({WINDOW_MODE}) |")
    a(f"| Time in elevation window | **{times.along_track_s:.1f} s** |")
    a(f"| Extra time beyond the 30 deg stop (target still in the {optics.fov_el_deg:.2f} deg FOV) | {times.residual_s:.1f} s |")
    a(f"| Staring at nadir (no gimbal) | **{times.stare_s:.2f} s** |")
    a(f"| Peak elevation rate | {times.peak_el_rate_deg_s:.3f} deg/s |")
    a(f"| Mission slew cap | {MAX_SLEW_DEG_S:.1f} deg/s |")
    a(f"| Time from CA of window start / stop | {times.t_start_s:.1f} / {times.t_stop_s:.1f} s |")
    a(f"| Slant range at start / stop | {times.slant_start_km:.1f} / {times.slant_stop_km:.1f} km |")
    a("")
    if times_420 is not None and orbit_420 is not None:
        a(f"At {orbit_420.altitude_km:.0f} km altitude the one-sided elevation window is **{times_420.along_track_s:.1f} s**")
        a(f"(stare {times_420.stare_s:.2f} s). Altitude in the ISS band changes dwell by a few percent.")
        a("")
    if times_two_sided is not None:
        ratio = times_two_sided.along_track_s / times.along_track_s
        a(
            f"A two-sided elevation axis (through nadir to {EL_LIMB_DEG:.0f} deg aft) "
            f"gives **{times_two_sided.along_track_s:.1f} s** of along-track track "
            f"({ratio:.2f}× the one-sided window), still with no azimuth motor."
        )
        a("")
    a("Near nadir the target races through the 2.7 deg along-track FOV in a couple")
    a("of seconds. That is why the elevation axis is the one that matters for SWaP:")
    a(f"it turns ~{times.stare_s:.1f} s of staring time into **{times.along_track_s:.0f} s** of track.")
    a("")
    a("## Lateral loss of dropping the azimuth axis")
    a("")
    a("The CoG of a ground-fixed disk subtends the largest azimuth at closest")
    a("approach, where range is shortest. If that peak azimuth stays inside the")
    a("sensor half-FOV, a one-axis gimbal parked at az = 0 never loses the CoG.")
    a("")
    a(f"| R (km) | peak \\|az\\| (deg) | 1-axis time (s) | 2-axis time (s) | time lost (s) | lost fraction |")
    a("| --- | --- | --- | --- | --- | --- |")
    for i, r in enumerate(table["radius_km"]):
        t1 = float(table["one_axis_s"][i])
        t2 = float(table["two_axis_s"][i])
        lost = float(table["lost_s"][i])
        frac = 0.0 if t2 <= 0 else lost / t2
        a(
            f"| {r:.1f} | {table['az_peak_deg'][i]:.2f} | {t1:.1f} | {t2:.1f} | "
            f"{lost:.1f} | {100 * frac:.1f}% |"
        )
    a("")
    a(f"Nadir fit: a disk with **R ≤ {r_fit:.2f} km** has peak \\|az\\| ≤ sensor")
    a("half-FOV. For any such R, **one-axis and two-axis tracking times are the")
    a("same** (both are elevation-window limited).")
    a("")
    a("Industrial smoke plumes in the Mommert Sentinel-2 corpus are labelled in")
    a("1.2 km tiles; published visible industrial plumes are often a few hundred")
    a("metres to a few km. Power-plant SO2 plumes have been traversed at 4–16 km")
    a("downwind, so a *long* plume's centroid can sit several km from the stack.")
    a("Even a conservative **R = 5–10 km** is still inside the 1-axis nadir swath.")
    a("")
    a("### Superposition of several plumes")
    a("")
    a("If plume *i* has CoG in a disk of radius `r_i` around origin `o_i`, and the")
    a("gimbal tracks the weighted centroid `c = Σ w_i p_i` (`w_i ≥ 0`, `Σ w_i = 1`):")
    a("")
    a("- `c` stays in a disk of radius **`R_centroid = Σ w_i r_i`** around `Σ w_i o_i`.")
    a("- Equal-radius plumes: `R_centroid = r`, **independent of N**.")
    a("- N does **not** add the radii unless the tracker is allowed to put all of")
    a("  the weight on one plume (i.e. it switches to a single plume). That case is")
    a("  `R = r` around *that* stack, and the stacks' lateral separation is an")
    a("  origin offset, not a bigger disk about one origin.")
    a("- To keep *every* plume inside the frame, use the covering radius")
    a("  `R_cover = max_i (|o_i - o_c| + r_i) = D + r` for origins in a cluster")
    a("  of half-span D.")
    a("")
    a("So 'superposition of the radii' is `Σ w_i r_i` for centroid tracking, and")
    a("`D + r` for covering a cluster. It is not `N r` unless the origins themselves")
    a("are allowed to sit `r` apart in a chain and we must cover all of them.")
    a("")
    a("### Wind during the pass")
    a("")
    a("A CoG that advects at wind speed `v` during the elevation window picks up")
    a(f"an extra ground displacement `v × {times.along_track_s:.0f} s`.")
    a("")
    a("| wind (m/s) | extra displacement (km) | leftover 1-axis budget (km) at nadir |")
    a("| --- | --- | --- |")
    for w, extra, left in wind_rows:
        a(f"| {w:.0f} | {extra:.2f} | {left:.2f} |")
    a("")
    a("## If the original analysis was 'anywhere in the ±10 deg box'")
    a("")
    a("That envelope is a **pointing** box, not the sensor FOV. At nadir it is a")
    a(f"**{2 * r_box:.0f} km** wide ground strip. The 150 mm sensor only sees")
    a(f"**{2 * r_fit:.1f} km** of that strip. A two-axis gimbal is what lets you")
    a("place the 3.2 deg chip anywhere in the 20 deg box.")
    a("")
    a("A stack that sits 40–70 km off the slice centre is inside the two-axis box")
    a("and **outside** the one-axis chip for the whole pass — 100% of that")
    a("target's tracking time is lost. That is the loss if we still needed to")
    a("cover the full ±10 deg operational slice.")
    a("")
    a("| origin cross-track (km) | 1-axis time (s) | 2-axis time (s) | lost (s) |")
    a("| --- | --- | --- | --- |")
    for y, t1, t2 in offsets:
        a(f"| {y:.0f} | {t1:.1f} | {t2:.1f} | {t2 - t1:.1f} |")
    a("")
    a("Under the new model (one origin, CoG in a disk of a few km, slice already")
    a("aligned with that origin) that off-axis population is not in the")
    a("requirement, and the azimuth axis buys **no extra tracking time**.")
    a("")
    a("## Bottom line (under the defaults above)")
    a("")
    a(f"1. Along-track, a single elevation axis gives **{times.along_track_s:.0f} s**")
    a(f"   of track (vs **{times.stare_s:.1f} s** staring). Peak rate")
    a(f"   {times.peak_el_rate_deg_s:.2f} deg/s is under the {MAX_SLEW_DEG_S:.0f} deg/s cap.")
    a(f"2. Lateral, the 2448 px axis sees ±{optics.half_az_deg:.2f} deg ≈")
    a(f"   **±{r_fit:.1f} km** at nadir. For R ≲ {r_fit:.0f} km the CoG never")
    a("   leaves the chip, so dropping azimuth costs **0 s** of tracking time.")
    a("3. The azimuth axis only pays for itself if we must (a) keep a CoG on")
    a("   boresight rather than merely in-frame, (b) cover origins tens of km")
    a("   off the slice centre, or (c) cover a cluster with D + r ≳ 11 km.")
    a("4. A two-sided elevation axis (through nadir) would roughly **double**")
    a("   along-track time without adding the second motor.")
    a("")
    a("## Figures")
    a("")
    a("- `outputs/geometry_overview.png` — elevation window vs sensor FOV")
    a("- `outputs/along_track_timeline.png` — elevation, slew rate, slant range")
    a("- `outputs/disk_angular_radius.png` — worst-case az of a ground disk vs time")
    a("- `outputs/tracking_time_vs_radius.png` — 1-axis vs 2-axis vs R")
    a("- `outputs/lost_time_vs_radius.png` — time given up by dropping azimuth")
    a("- `outputs/footprint_vs_time.png` — ground half-width of the chip")
    a("- `outputs/origin_offset.png` — stack not on the slice centreline")
    a("- `outputs/required_az_vs_radius.png` — az travel a 2-axis unit would use")
    a("- `outputs/tracking_time_vs_radius.csv`, `outputs/origin_offset.csv` — tables")
    a("")
    a("## How to regenerate")
    a("")
    a("```text")
    a("uv run python scratch/single-axis-gimbal-analysis/analyze.py")
    a("```")
    a("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_check(orbit: Orbit, optics: Optics, times: WindowTimes) -> None:
    s0, v0 = iss_state(0.0, orbit)
    look0 = look_at(s0, v0, ground_point(0.0, 0.0))
    assert look0.visible
    assert abs(look0.el_deg - 90.0) < 0.05, look0.el_deg
    assert abs(look0.az_deg) < 0.05, look0.az_deg
    # A purely cross-track target at CA is an az offset ≈ atan(y/h).
    y = 10.0
    look_y = look_at(s0, v0, ground_point(0.0, y))
    expected = math.degrees(math.atan(y / orbit.altitude_km))
    assert abs(abs(look_y.cross_off_nadir_deg) - expected) < 0.2, (
        look_y.cross_off_nadir_deg,
        expected,
    )
    # Approaching: t<0, el between 30 and 90.
    assert times.t_start_s < 0.0 < times.t_stop_s or times.t_stop_s <= 0.0
    # For a centerline origin the one-sided window should end at CA (el=90)
    # and start at el=30 on the approach.
    assert times.along_track_s > 60.0
    assert times.along_track_s < 180.0
    assert times.peak_el_rate_deg_s < MAX_SLEW_DEG_S
    # Datasheet 2/3" HFOV should match 3.36 deg within 0.02 deg.
    assert abs(optics.computed_2_3_hfov_deg - DATASHEET_HFOV_2_3_DEG) < 0.02
    print("self-check: ok")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    orbit = build_orbit(ALTITUDE_KM)
    optics = build_optics(orbit.altitude_km)
    times, data = origin_window(orbit, optics, 0.0, WINDOW_MODE)
    self_check(orbit, optics, times)
    table = tracking_time_vs_radius(orbit, optics, 0.0, DISK_RADII_KM, WINDOW_MODE)
    orbit_420 = build_orbit(420.0)
    optics_420 = build_optics(orbit_420.altitude_km)
    times_420, _ = origin_window(orbit_420, optics_420, 0.0, WINDOW_MODE)
    times_two, _ = origin_window(orbit, optics, 0.0, "two_sided")

    offsets = offset_times(orbit, optics, WINDOW_MODE)
    plot_geometry(orbit, optics, OUT / "geometry_overview.png")
    plot_along_track(orbit, optics, times, data, OUT / "along_track_timeline.png")
    plot_disk_angle(orbit, optics, times, OUT / "disk_angular_radius.png")
    plot_time_vs_radius(table, optics, OUT / "tracking_time_vs_radius.png")
    plot_lost_time(table, OUT / "lost_time_vs_radius.png")
    plot_footprint(orbit, optics, times, OUT / "footprint_vs_time.png")
    plot_offset_map(offsets, OUT / "origin_offset.png")
    plot_required_az(table, optics, OUT / "required_az_vs_radius.png")

    # CSV
    csv_path = OUT / "tracking_time_vs_radius.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("radius_km,az_peak_deg,one_axis_s,two_axis_s,lost_s\n")
        for i, r in enumerate(table["radius_km"]):
            fh.write(
                f"{r:.3f},{table['az_peak_deg'][i]:.4f},"
                f"{table['one_axis_s'][i]:.3f},{table['two_axis_s'][i]:.3f},"
                f"{table['lost_s'][i]:.3f}\n"
            )
    off_path = OUT / "origin_offset.csv"
    with off_path.open("w", encoding="utf-8") as fh:
        fh.write("origin_cross_km,one_axis_s,two_axis_s,lost_s\n")
        for y, t1, t2 in offsets:
            fh.write(f"{y:.1f},{t1:.3f},{t2:.3f},{t2 - t1:.3f}\n")

    write_report(
        orbit,
        optics,
        times,
        table,
        offsets,
        orbit_420,
        times_420,
        times_two,
        ROOT / "RESULTS.md",
    )

    print()
    print(f"altitude          {orbit.altitude_km:.1f} km")
    print(f"FOV az x el       {optics.fov_az_deg:.3f} x {optics.fov_el_deg:.3f} deg")
    print(f"nadir GSD         {optics.gsd_nadir_m:.2f} m")
    print(f"along-track time  {times.along_track_s:.1f} s")
    print(f"stare time         {times.stare_s:.2f} s")
    print(f"peak el rate       {times.peak_el_rate_deg_s:.3f} deg/s")
    print(f"nadir half-swath  {orbit.altitude_km * math.tan(math.radians(optics.half_az_deg)):.2f} km")
    print(f"wrote {OUT}")
    print(f"wrote {ROOT / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
