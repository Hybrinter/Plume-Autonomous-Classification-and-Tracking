"""Pass sampling, science-window masks, and T(lat, R) interpolation.

A pass is sampled in ECI with Earth rotation applied to the ECEF target.
One-axis tracking keeps any part of the unknown-wind covering disk inside
the parked sensor FOV; two-axis points at a disk point inside the azimuth
keep-out box (the chip then contains that point). ``TimeLostFn`` caches a
(|lat|, R) grid and interpolates tracking time.

Contains:
  - SampleSpan / PassSamples: time grid and look-angle arrays.
  - sample_pass / in_elevation_window / in_science_window / mask_time_s.
  - angular_rate_deg_s / two_axis_boresightable / staring_in_frame.
  - az_interval_overlaps / disk_any_part_in_stop.
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


def az_interval_overlaps(az_a: np.ndarray, az_b: np.ndarray, half_deg: float) -> np.ndarray:
    """Return whether [min(az_a, az_b), max(az_a, az_b)] overlaps [-half, half].

    Args:
        az_a: First azimuth samples, degrees. # np.ndarray[float64, (N,)]
        az_b: Second azimuth samples, degrees. # np.ndarray[float64, (N,)]
        half_deg: Symmetric stop in degrees.

    Returns:
        Boolean mask. # np.ndarray[bool, (N,)]
    """
    lo = np.minimum(az_a, az_b)
    hi = np.maximum(az_a, az_b)
    return cast(np.ndarray, (lo <= half_deg) & (hi >= -half_deg))


def disk_any_part_in_stop(
    az_origin: np.ndarray,
    az_plus: np.ndarray,
    az_minus: np.ndarray,
    half_deg: float,
) -> np.ndarray:
    """Return whether any part of the covering disk is inside +/- ``half_deg``.

    The disk azimuth span is the interval covering the origin and both
    cross-track edges. Unknown wind puts the plume CoG somewhere in that
    span; this is the any-part (existence) test, not whole-disk containment.

    Args:
        az_origin: Origin azimuth samples, degrees. # np.ndarray[float64, (N,)]
        az_plus: +R cross-track edge azimuth. # np.ndarray[float64, (N,)]
        az_minus: -R cross-track edge azimuth. # np.ndarray[float64, (N,)]
        half_deg: Symmetric stop in degrees (FOV half-az or keep-out box).

    Returns:
        Boolean mask. # np.ndarray[bool, (N,)]
    """
    lo = np.minimum(np.minimum(az_origin, az_plus), az_minus)
    hi = np.maximum(np.maximum(az_origin, az_plus), az_minus)
    return cast(np.ndarray, (lo <= half_deg) & (hi >= -half_deg))


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

    The grid is built by sampling the origin and the +/- R disk edges through
    the science window. Earth rotation is on. One-axis in-frame means any
    part of the disk overlaps the parked FOV. Two-axis in-frame means any
    part of the disk overlaps the azimuth keep-out box (then boresight that
    point). Values outside the grid clip to the inclination in latitude and
    to the radius bounds.

    Args:
        orbit: Circular ISS orbit.
        optics: Usable sensor FOV (one-axis azimuth stop = half_az).
        box: Gimbal box (two-axis azimuth stop and elevation window).
        lats_deg: Grid latitudes in degrees, typically 0 to inclination.
        radii_km: Covering-disk radii in kilometres.
        cache_path: ``.npz`` cache. Rebuilt if latitudes, radii, or box
            stops differ.
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
            same_grid = np.allclose(cached["lats"], self.lats) and np.allclose(
                cached["rs"], self.rs
            )
            same_box = _cap_matches(
                _npz_float(cached, "el_limb_deg"), box.el_limb_deg
            ) and _cap_matches(_npz_float(cached, "az_box_deg"), box.az_box_deg)
            if same_grid and same_box:
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
                    edge_p = sample_pass(orbit, box, float(lat), 0.0, float(radius), self._span)
                    edge_m = sample_pass(orbit, box, float(lat), 0.0, -float(radius), self._span)
                    one = science & disk_any_part_in_stop(
                        origin.az, edge_p.az, edge_m.az, optics.half_az_deg
                    )
                    two = science & disk_any_part_in_stop(
                        origin.az, edge_p.az, edge_m.az, box.az_box_deg
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
