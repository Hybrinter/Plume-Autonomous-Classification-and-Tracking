"""Pass sampling, science-window masks, and T(lat, R) interpolation.

A pass is sampled in ECI with Earth rotation applied to the ECEF target.
In-frame means at least half the covering-disk area is on the sensor chip
after pointing. One-axis parks at az = 0 (elevation tracks the origin).
Two-axis boresights the origin when it is inside the azimuth keep-out box.
``TimeLostFn`` caches a (|lat|, R) grid and interpolates tracking time.

Contains:
  - SampleSpan / PassSamples: time grid and look-angle arrays.
  - sample_pass / in_elevation_window / in_science_window / mask_time_s.
  - angular_rate_deg_s / two_axis_boresightable / staring_in_frame.
  - disk_frac_in_rect / disk_half_in_chip.
  - TimeLostFn: interpolator for one-axis and two-axis tracking time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from analysis.lib.constants import OMEGA_EARTH_RAD_S
from analysis.lib.look import GimbalBox, WindowMode, look_at, rotate_z
from analysis.lib.optics import Optics
from analysis.lib.orbit import (
    Orbit,
    argument_of_latitude,
    heading_from_north_deg,
    iss_eci,
    offset_ecef,
    origin_ecef,
)


@dataclass(frozen=True)
class SampleSpan:
    """Time grid for one pass sample.

    Attributes:
        t_min_s: First sample, seconds from closest approach (negative before).
        t_max_s: Last sample, seconds from closest approach.
        dt_s: Sample step in seconds.
    """

    t_min_s: float = -250.0
    t_max_s: float = 50.0
    dt_s: float = 0.05


@dataclass
class PassSamples:
    """Look-angle time series for one origin or disk-edge sample.

    Arrays are float64 length N except ``vis`` (bool).

    Attributes:
        t: Seconds from closest approach. # np.ndarray[float64, (N,)]
        az: Azimuth degrees. # np.ndarray[float64, (N,)]
        el: Elevation degrees. # np.ndarray[float64, (N,)]
        eta: Off-nadir degrees. # np.ndarray[float64, (N,)]
        slant: Slant range kilometres. # np.ndarray[float64, (N,)]
        incidence: Earth emission angle degrees. # np.ndarray[float64, (N,)]
        vis: Earth-hit flag. # np.ndarray[bool, (N,)]
    """

    t: np.ndarray
    az: np.ndarray
    el: np.ndarray
    eta: np.ndarray
    slant: np.ndarray
    incidence: np.ndarray
    vis: np.ndarray


def sample_pass(
    orbit: Orbit,
    box: GimbalBox,
    lat_deg: float,
    along_km: float,
    cross_km: float,
    span: SampleSpan | None = None,
) -> PassSamples:
    """Sample look angles to a surface point through one ISS pass.

    Earth rotation is applied to the ECEF target. t = 0 is the
    northern-hemisphere sub-satellite point at ``lat_deg``.

    Args:
        orbit: Circular ISS orbit.
        box: Gimbal box (only ``el_nadir_deg`` is used here).
        lat_deg: Pass geocentric latitude in degrees.
        along_km: Target along-track offset from the sub-satellite origin.
        cross_km: Target cross-track offset from the sub-satellite origin.
        span: Time grid. Default is -250 s to +50 s at 0.05 s.

    Returns:
        Look-angle samples. Arrays share one length N.
    """
    used = span if span is not None else SampleSpan()
    u0 = argument_of_latitude(lat_deg, orbit.inclination_rad)
    heading = heading_from_north_deg(lat_deg, orbit.inclination_rad)
    origin = origin_ecef(orbit, lat_deg)
    tgt_ecef = offset_ecef(origin, lat_deg, along_km, cross_km, heading)
    ts = np.arange(used.t_min_s, used.t_max_s + used.dt_s * 0.5, used.dt_s)
    n_samp = ts.size
    az = np.empty(n_samp)
    el = np.empty(n_samp)
    eta = np.empty(n_samp)
    slant = np.empty(n_samp)
    incidence = np.empty(n_samp)
    vis = np.empty(n_samp, dtype=bool)
    for i, t_s in enumerate(ts):
        r_iss, vel = iss_eci(float(t_s), orbit, u0)
        tgt = rotate_z(OMEGA_EARTH_RAD_S * float(t_s)) @ tgt_ecef
        look = look_at(r_iss, vel, tgt, box.el_nadir_deg)
        az[i] = look.az_deg
        el[i] = look.el_deg
        eta[i] = look.eta_deg
        slant[i] = look.slant_km
        incidence[i] = look.incidence_deg
        vis[i] = look.visible
    return PassSamples(t=ts, az=az, el=el, eta=eta, slant=slant, incidence=incidence, vis=vis)


def in_elevation_window(el_deg: np.ndarray, box: GimbalBox) -> np.ndarray:
    """Return a mask of samples inside the elevation window.

    Args:
        el_deg: Elevation samples in degrees. # np.ndarray[float64, (N,)]
        box: Gimbal box (nadir, limb, window mode).

    Returns:
        Boolean mask. # np.ndarray[bool, (N,)]
    """
    forward = (el_deg >= box.el_limb_deg) & (el_deg <= box.el_nadir_deg)
    if box.window_mode is WindowMode.ONE_SIDED:
        return forward
    half = box.el_nadir_deg - box.el_limb_deg
    return np.abs(el_deg - box.el_nadir_deg) <= half


def in_science_window(samples: PassSamples, box: GimbalBox) -> np.ndarray:
    """Return a mask of samples inside the science elevation window.

    The off-nadir stop is ``box.el_limb_deg`` (eta_max from along-track
    band GSD). Slant range and incidence are not extra caps.

    Args:
        samples: Origin look-angle samples.
        box: Gimbal box (elevation window).

    Returns:
        Boolean mask. # np.ndarray[bool, (N,)]
    """
    return cast(np.ndarray, in_elevation_window(samples.el, box) & samples.vis)


DISK_AREA_FRAC_MIN = 0.5


def _area_coord_ge(radius_km: np.ndarray, height_km: np.ndarray) -> np.ndarray:
    """Return disk area with a coordinate >= ``height_km``.

    The disk is radius ``radius_km`` centered at the origin. The formula is
    the circular segment (half-plane) area r^2 acos(h/r) - h sqrt(r^2-h^2).

    Args:
        radius_km: Disk radii, kilometres. # np.ndarray[float64, (...)]
        height_km: Half-plane edge, kilometres. # np.ndarray[float64, (...)]

    Returns:
        Area in km2, broadcast of the inputs. # np.ndarray[float64, (...)]
    """
    radius = np.maximum(radius_km, 0.0)
    full = math.pi * radius * radius
    ratio = np.clip(height_km / np.maximum(radius, 1.0e-15), -1.0, 1.0)
    root = np.sqrt(np.maximum(radius * radius - height_km * height_km, 0.0))
    mid = radius * radius * np.arccos(ratio) - height_km * root
    result = np.where(height_km >= radius, 0.0, np.where(height_km <= -radius, full, mid))
    return np.where(radius <= 0.0, 0.0, result)


def _g_first_quad(radius_km: np.ndarray, x_km: np.ndarray, y_km: np.ndarray) -> np.ndarray:
    """Return disk area in [0, x] x [0, y] for x >= 0, y >= 0.

    The disk is at the origin. Arguments are clipped to the first quadrant
    and to the disk bounding box.

    Args:
        radius_km: Disk radii, kilometres. # np.ndarray[float64, (...)]
        x_km: Far x edge, kilometres. # np.ndarray[float64, (...)]
        y_km: Far y edge, kilometres. # np.ndarray[float64, (...)]

    Returns:
        Area in km2. # np.ndarray[float64, (...)]
    """
    radius = np.maximum(radius_km, 0.0)
    x_pos = np.minimum(np.maximum(x_km, 0.0), radius)
    y_pos = np.minimum(np.maximum(y_km, 0.0), radius)
    inside = x_pos * x_pos + y_pos * y_pos <= radius * radius
    quarter = 0.25 * math.pi * radius * radius
    outside_area = (
        quarter - 0.5 * _area_coord_ge(radius, x_pos) - 0.5 * _area_coord_ge(radius, y_pos)
    )
    area = np.where(inside, x_pos * y_pos, outside_area)
    return np.where(radius <= 0.0, 0.0, np.maximum(area, 0.0))


def _disk_q1_rect(
    radius_km: np.ndarray,
    xmin_km: np.ndarray,
    xmax_km: np.ndarray,
    ymin_km: np.ndarray,
    ymax_km: np.ndarray,
) -> np.ndarray:
    """Return disk area in the first-quadrant part of an axis-aligned rect.

    Args:
        radius_km: Disk radii, kilometres. # np.ndarray[float64, (...)]
        xmin_km: Left edge, kilometres. # np.ndarray[float64, (...)]
        xmax_km: Right edge, kilometres. # np.ndarray[float64, (...)]
        ymin_km: Bottom edge, kilometres. # np.ndarray[float64, (...)]
        ymax_km: Top edge, kilometres. # np.ndarray[float64, (...)]

    Returns:
        Area in km2. # np.ndarray[float64, (...)]
    """
    x0 = np.maximum(xmin_km, 0.0)
    x1 = np.maximum(xmax_km, 0.0)
    y0 = np.maximum(ymin_km, 0.0)
    y1 = np.maximum(ymax_km, 0.0)
    return cast(
        np.ndarray,
        _g_first_quad(radius_km, x1, y1)
        - _g_first_quad(radius_km, x0, y1)
        - _g_first_quad(radius_km, x1, y0)
        + _g_first_quad(radius_km, x0, y0),
    )


def disk_frac_in_rect(
    radius_km: float | np.ndarray,
    cx_km: float | np.ndarray,
    cy_km: float | np.ndarray,
    xmin_km: float | np.ndarray,
    xmax_km: float | np.ndarray,
    ymin_km: float | np.ndarray,
    ymax_km: float | np.ndarray,
) -> np.ndarray:
    """Return the fraction of a covering disk inside an axis-aligned rectangle.

    The disk is radius ``radius_km`` centered at (cx, cy). A zero-radius disk
    is a point: the fraction is 1 when the point is inside the rectangle.

    Args:
        radius_km: Covering radius, kilometres.
        cx_km: Disk center x, kilometres (along-track).
        cy_km: Disk center y, kilometres (cross-track).
        xmin_km: Rectangle left edge, kilometres.
        xmax_km: Rectangle right edge, kilometres.
        ymin_km: Rectangle bottom edge, kilometres.
        ymax_km: Rectangle top edge, kilometres.

    Returns:
        Fraction in [0, 1], broadcast of the inputs. # np.ndarray[float64, (...)]
    """
    radius = np.asarray(radius_km, dtype=float)
    cx = np.asarray(cx_km, dtype=float)
    cy = np.asarray(cy_km, dtype=float)
    xmin = np.asarray(xmin_km, dtype=float) - cx
    xmax = np.asarray(xmax_km, dtype=float) - cx
    ymin = np.asarray(ymin_km, dtype=float) - cy
    ymax = np.asarray(ymax_km, dtype=float) - cy
    radius, xmin, xmax, ymin, ymax = np.broadcast_arrays(radius, xmin, xmax, ymin, ymax)
    area = (
        _disk_q1_rect(radius, xmin, xmax, ymin, ymax)
        + _disk_q1_rect(radius, -xmax, -xmin, ymin, ymax)
        + _disk_q1_rect(radius, -xmax, -xmin, -ymax, -ymin)
        + _disk_q1_rect(radius, xmin, xmax, -ymax, -ymin)
    )
    full = math.pi * np.maximum(radius, 0.0) ** 2
    point_in = (xmin <= 0.0) & (0.0 <= xmax) & (ymin <= 0.0) & (0.0 <= ymax)
    frac = np.divide(area, full, out=np.zeros(full.shape, dtype=float), where=full > 0.0)
    frac = np.where(full > 0.0, frac, point_in.astype(float))
    empty = (xmax <= xmin) | (ymax <= ymin)
    return np.clip(np.where(empty, 0.0, frac), 0.0, 1.0)


def disk_half_in_chip(
    az_deg: np.ndarray,
    slant_km: np.ndarray,
    radius_km: float,
    optics: Optics,
    *,
    boresight_origin: bool,
    az_box_deg: float | None = None,
) -> np.ndarray:
    """Return whether at least half the covering disk is on the chip.

    The chip is a local-flat ground rectangle of half-widths
    slant * tan(half_fov). One-axis parks at az = 0, so the origin sits at
    cross-track slant * tan(az). Two-axis boresights the origin (circle at
    the chip center) when ``|az| <= az_box_deg``.

    Args:
        az_deg: Origin azimuth samples, degrees. # np.ndarray[float64, (N,)]
        slant_km: Slant range samples, kilometres. # np.ndarray[float64, (N,)]
        radius_km: Covering-disk radius, kilometres.
        optics: Usable sensor FOV.
        boresight_origin: True for two-axis (chip centered on the origin).
        az_box_deg: Two-axis keep-out half-angle. Required when boresighting.

    Returns:
        Boolean mask. # np.ndarray[bool, (N,)]

    Raises:
        ValueError: If ``boresight_origin`` is True and ``az_box_deg`` is None.
    """
    if boresight_origin and az_box_deg is None:
        raise ValueError("az_box_deg is required when boresight_origin is True")
    w_az = slant_km * np.tan(np.radians(optics.half_az_deg))
    w_el = slant_km * np.tan(np.radians(optics.half_el_deg))
    if boresight_origin:
        y_c = np.zeros_like(az_deg, dtype=float)
    else:
        y_c = slant_km * np.tan(np.radians(az_deg))
    frac = disk_frac_in_rect(radius_km, 0.0, y_c, -w_el, w_el, -w_az, w_az)
    mask = frac >= DISK_AREA_FRAC_MIN
    if boresight_origin:
        assert az_box_deg is not None
        mask = mask & (np.abs(az_deg) <= az_box_deg)
    return cast(np.ndarray, mask)


def mask_time_s(mask: np.ndarray, t_s: np.ndarray) -> float:
    """Return the duration implied by a boolean mask on a uniform time grid.

    Args:
        mask: Samples to count. # np.ndarray[bool, (N,)]
        t_s: Time samples in seconds. # np.ndarray[float64, (N,)]

    Returns:
        Duration in seconds (sum of mask times median dt). Zero if empty.
    """
    if t_s.size < 2 or not np.any(mask):
        return 0.0
    dt_s = float(np.median(np.diff(t_s)))
    return float(np.sum(mask) * dt_s)


def angular_rate_deg_s(t_s: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    """Return d(angle)/dt on the sample grid.

    Args:
        t_s: Time samples in seconds. # np.ndarray[float64, (N,)]
        angle_deg: Angle samples in degrees. # np.ndarray[float64, (N,)]

    Returns:
        Angular rate in degrees per second. # np.ndarray[float64, (N,)]
    """
    return cast(np.ndarray, np.gradient(angle_deg, t_s))


def two_axis_boresightable(az_deg: np.ndarray, el_deg: np.ndarray, box: GimbalBox) -> np.ndarray:
    """Return a mask of samples the two-axis box can boresight.

    Args:
        az_deg: Azimuth samples in degrees. # np.ndarray[float64, (N,)]
        el_deg: Elevation samples in degrees. # np.ndarray[float64, (N,)]
        box: Gimbal box.

    Returns:
        Boolean mask. # np.ndarray[bool, (N,)]
    """
    return cast(np.ndarray, in_elevation_window(el_deg, box) & (np.abs(az_deg) <= box.az_box_deg))


def staring_in_frame(
    az_deg: np.ndarray, el_deg: np.ndarray, optics: Optics, box: GimbalBox
) -> np.ndarray:
    """Return a mask of samples inside the staring (no-slew) sensor FOV.

    Args:
        az_deg: Azimuth samples in degrees. # np.ndarray[float64, (N,)]
        el_deg: Elevation samples in degrees. # np.ndarray[float64, (N,)]
        optics: Usable sensor FOV.
        box: Gimbal box (nadir elevation).

    Returns:
        Boolean mask. # np.ndarray[bool, (N,)]
    """
    return cast(
        np.ndarray,
        (np.abs(az_deg) <= optics.half_az_deg)
        & (np.abs(el_deg - box.el_nadir_deg) <= optics.half_el_deg),
    )


class TimeLostFn:
    """Interpolate one-axis and two-axis tracking time vs |lat| and covering R.

    The grid samples the origin through the science window. Earth rotation is
    on. In-frame means at least half the covering-disk area is on the chip
    after pointing. One-axis parks at az = 0. Two-axis boresights the origin
    when it is inside the azimuth keep-out box. Values outside the grid clip
    to the inclination in latitude and to the radius bounds.

    Args:
        orbit: Circular ISS orbit.
        optics: Usable sensor FOV (chip half-angles).
        box: Gimbal box (two-axis azimuth stop and elevation window).
        lats_deg: Grid latitudes in degrees, typically 0 to inclination.
        radii_km: Covering-disk radii in kilometres.
        cache_path: ``.npz`` cache. Rebuilt if latitudes, radii, box stops,
            or the area-fraction gate differ.
        span: Pass sample grid. Default dt is 0.1 s for the industry grid.
        verbose: Print per-latitude progress when rebuilding the cache.
    """

    def __init__(
        self,
        orbit: Orbit,
        optics: Optics,
        box: GimbalBox,
        lats_deg: np.ndarray,
        radii_km: np.ndarray,
        cache_path: Path,
        *,
        span: SampleSpan | None = None,
        verbose: bool = False,
    ) -> None:
        self.orbit = orbit
        self.optics = optics
        self.box = box
        self.lats = np.asarray(lats_deg, dtype=float)
        self.rs = np.asarray(radii_km, dtype=float)
        self._span = span if span is not None else SampleSpan(dt_s=0.1)
        nlat, nr = self.lats.size, self.rs.size
        t1: np.ndarray | None = None
        t2: np.ndarray | None = None
        if cache_path.is_file():
            cached = np.load(cache_path)
            same_grid = (
                cached["lats"].shape == self.lats.shape
                and cached["rs"].shape == self.rs.shape
                and np.allclose(cached["lats"], self.lats)
                and np.allclose(cached["rs"], self.rs)
            )
            same_box = _cap_matches(
                _npz_float(cached, "el_limb_deg"), box.el_limb_deg
            ) and _cap_matches(_npz_float(cached, "az_box_deg"), box.az_box_deg)
            same_area = _cap_matches(_npz_float(cached, "area_frac"), DISK_AREA_FRAC_MIN)
            if same_grid and same_box and same_area:
                t1 = cached["t1"]
                t2 = cached["t2"]
                if verbose:
                    print(f"  time-lost grid cache {cache_path}")
        if t1 is None or t2 is None:
            t1 = np.zeros((nlat, nr))
            t2 = np.zeros((nlat, nr))
            if verbose:
                print(f"building time-lost grid {nlat} lats x {nr} radii, dt={self._span.dt_s}s")
            for i, lat in enumerate(self.lats):
                origin = sample_pass(orbit, box, float(lat), 0.0, 0.0, self._span)
                science = in_science_window(origin, box)
                for j, radius in enumerate(self.rs):
                    one = science & disk_half_in_chip(
                        origin.az,
                        origin.slant,
                        float(radius),
                        optics,
                        boresight_origin=False,
                    )
                    two = science & disk_half_in_chip(
                        origin.az,
                        origin.slant,
                        float(radius),
                        optics,
                        boresight_origin=True,
                        az_box_deg=box.az_box_deg,
                    )
                    t1[i, j] = mask_time_s(one, origin.t)
                    t2[i, j] = mask_time_s(two, origin.t)
                if verbose:
                    t1_lo = float(t1[i, 0])
                    print(f"  lat {lat:6.2f}  T1(Rmin)={t1_lo:6.1f}s  T2={t2[i, 0]:6.1f}s")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                lats=self.lats,
                rs=self.rs,
                t1=t1,
                t2=t2,
                el_limb_deg=box.el_limb_deg,
                az_box_deg=box.az_box_deg,
                area_frac=DISK_AREA_FRAC_MIN,
            )
        assert t1 is not None and t2 is not None
        self._t1 = RegularGridInterpolator(
            (self.lats, self.rs), t1, bounds_error=False, fill_value=None
        )
        self._t2 = RegularGridInterpolator(
            (self.lats, self.rs), t2, bounds_error=False, fill_value=None
        )
        self.t1_grid = t1
        self.t2_grid = t2

    def eval(self, lat_deg: float, radius_km: float) -> tuple[float, float, float]:
        """Return (T_one_axis, T_two_axis, lost) seconds at ``|lat|``, ``R``.

        Latitude is folded to |lat| and clipped to the orbit inclination.
        Radius is clipped to the grid bounds.

        Args:
            lat_deg: Pass latitude in degrees (sign ignored).
            radius_km: Covering-disk radius in kilometres.

        Returns:
            One-axis time, two-axis time, and two-axis minus one-axis.
        """
        lat_abs = min(self.orbit.inclination_deg, abs(float(lat_deg)))
        radius = float(np.clip(radius_km, float(self.rs[0]), float(self.rs[-1])))
        point = np.array([[lat_abs, radius]])
        t1 = float(self._t1(point)[0])
        t2 = float(self._t2(point)[0])
        if not math.isfinite(t1):
            t1 = 0.0
        if not math.isfinite(t2):
            t2 = 0.0
        return t1, t2, t2 - t1


def _npz_float(cached: np.lib.npyio.NpzFile, key: str) -> float | None:
    """Return a scalar from an npz, or None if the key is missing.

    Args:
        cached: Loaded npz archive.
        key: Array name.

    Returns:
        Scalar value, or None.
    """
    if key not in cached.files:
        return None
    return float(cached[key])


def _cap_matches(stored: float | None, current: float) -> bool:
    """Return True when ``stored`` equals ``current``, including inf.

    Args:
        stored: Value from cache, or None if missing.
        current: Expected scalar.

    Returns:
        True when both are inf or they match within 1e-6.
    """
    if stored is None:
        return False
    if math.isinf(stored) and math.isinf(current):
        return True
    return abs(stored - current) < 1e-6
