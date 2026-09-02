"""Latitude-binned plant span, covering radius, and hunt encounter rates.

Stack-weighted D(|lat|) comes from Climate TRACE clusters. Covering radius
is R = D + PLUME_R_KM. After a target leaves the frame the gimbal slews to
the elevation-window start and waits for the next cluster in the search
swath. Mean reacquire time is reset plus encounter; cycle time is single-
target dwell plus reacquire.

Contains:
  - LatBandRow / RadiusProfile.
  - build_radius_profile / folded_band_rows.
  - encounter_time_s / reacquire_s / cycle_s / ground_speed_km_s / hunt_at_lat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from analysis.lib.constants import MEAN_EARTH_RADIUS_KM
from analysis.lib.optics import Optics
from analysis.lib.orbit import Orbit
from analysis.studies.single_axis_vs_dual_axis_gimbal.assumptions import (
    FOLDED_LAT_BANDS,
    GIMBAL_BOX,
    LAT_BIN_DEG,
    MAX_HW_SLEW_DEG_S,
    PLUME_R_KM,
    TLE,
)
from analysis.studies.single_axis_vs_dual_axis_gimbal.inventory import Cluster


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted mean, or 0 if the weight sum is 0.

    Args:
        values: Samples.
        weights: Non-negative weights.

    Returns:
        Weighted mean.
    """
    w_sum = float(np.sum(weights))
    if w_sum <= 0:
        return 0.0
    return float(np.sum(values * weights) / w_sum)


@dataclass(frozen=True)
class LatBandRow:
    """Stack-weighted covering-radius stats for one |lat| band.

    Attributes:
        lat_lo: Band low edge in degrees.
        lat_hi: Band high edge in degrees.
        lat_mid: Band midpoint in degrees.
        n_clusters: Cluster count in the band (both hemispheres).
        n_stacks: Stack count in the band.
        mean_d_km: Stack-weighted mean plant span D.
        mean_r_km: Stack-weighted mean covering radius R = D + L.
        mean_n: Stack-weighted mean sources per cluster.
        frac_singleton_stacks: Fraction of stacks in n=1 clusters.
        d_char_n2_km: Stack-weighted 2 D / sqrt(n) for n>=2 clusters.
        dens_per_km2: Cluster count per square kilometre of the band.
    """

    lat_lo: float
    lat_hi: float
    lat_mid: float
    n_clusters: int
    n_stacks: int
    mean_d_km: float
    mean_r_km: float
    mean_n: float
    frac_singleton_stacks: float
    d_char_n2_km: float
    dens_per_km2: float


@dataclass(frozen=True)
class RadiusProfile:
    """Interpolators for R(|lat|) and cluster area density.

    Attributes:
        abs_lat: Bin midpoints in degrees, increasing. # np.ndarray[float64, (B,)]
        mean_r_km: Stack-weighted mean R at those midpoints.
        mean_d_km: Stack-weighted mean D at those midpoints.
        dens_per_km2: Cluster density at those midpoints.
        rows: One LatBandRow per bin, including empty bins.
    """

    abs_lat: np.ndarray
    mean_r_km: np.ndarray
    mean_d_km: np.ndarray
    dens_per_km2: np.ndarray
    rows: tuple[LatBandRow, ...]

    def r_km(self, lat_deg: float) -> float:
        """Return interpolated covering radius at ``|lat|``.

        Args:
            lat_deg: Geocentric latitude in degrees.

        Returns:
            R in kilometres, clipped to the inclination.
        """
        x = min(TLE.inclination_deg, abs(float(lat_deg)))
        return float(np.interp(x, self.abs_lat, self.mean_r_km))

    def dens_km2(self, lat_deg: float) -> float:
        """Return interpolated cluster density at ``|lat|``.

        Args:
            lat_deg: Geocentric latitude in degrees.

        Returns:
            Clusters per square kilometre.
        """
        x = min(TLE.inclination_deg, abs(float(lat_deg)))
        return float(np.interp(x, self.abs_lat, self.dens_per_km2))


def _band_area_km2(lat_lo: float, lat_hi: float) -> float:
    """Return spherical area of both-hemisphere latitude bands, km2.

    Args:
        lat_lo: Low |lat| edge in degrees.
        lat_hi: High |lat| edge in degrees.

    Returns:
        Area in square kilometres. Zero if hi <= lo.
    """
    lo = math.radians(max(0.0, lat_lo))
    hi = math.radians(max(lo, lat_hi))
    if hi <= lo:
        return 0.0
    radius = MEAN_EARTH_RADIUS_KM
    return 4.0 * math.pi * radius * radius * (math.sin(hi) - math.sin(lo))


def _band_row(iss: list[Cluster], lat_lo: float, lat_hi: float, *, closed_hi: bool) -> LatBandRow:
    """Return stats for clusters with lat_lo <= |lat| < lat_hi.

    Args:
        iss: ISS-belt clusters.
        lat_lo: Low |lat| edge in degrees.
        lat_hi: High |lat| edge in degrees.
        closed_hi: If True, include |lat| == lat_hi.

    Returns:
        One LatBandRow. Empty bands have zero means and density.
    """
    part: list[Cluster] = []
    for cluster in iss:
        abs_lat = abs(cluster.lat)
        if abs_lat < lat_lo:
            continue
        if closed_hi:
            if abs_lat > lat_hi:
                continue
        elif abs_lat >= lat_hi:
            continue
        part.append(cluster)
    mid = 0.5 * (lat_lo + lat_hi)
    area = _band_area_km2(lat_lo, lat_hi)
    if not part:
        return LatBandRow(
            lat_lo=lat_lo,
            lat_hi=lat_hi,
            lat_mid=mid,
            n_clusters=0,
            n_stacks=0,
            mean_d_km=0.0,
            mean_r_km=PLUME_R_KM,
            mean_n=0.0,
            frac_singleton_stacks=0.0,
            d_char_n2_km=0.0,
            dens_per_km2=0.0,
        )
    w = np.array([float(c.n) for c in part])
    d_arr = np.array([c.r_plant_km for c in part])
    r_arr = np.array([c.r_cover_km for c in part])
    n_arr = np.array([float(c.n) for c in part])
    n_stacks = int(np.sum(w))
    n1 = float(np.sum(w[n_arr == 1.0]))
    n2_mask = n_arr >= 2.0
    if np.any(n2_mask):
        d_char = 2.0 * d_arr[n2_mask] / np.sqrt(n_arr[n2_mask])
        d_char_mean = _weighted_mean(d_char, w[n2_mask])
    else:
        d_char_mean = 0.0
    dens = float(len(part) / area) if area > 0.0 else 0.0
    return LatBandRow(
        lat_lo=lat_lo,
        lat_hi=lat_hi,
        lat_mid=mid,
        n_clusters=len(part),
        n_stacks=n_stacks,
        mean_d_km=_weighted_mean(d_arr, w),
        mean_r_km=_weighted_mean(r_arr, w),
        mean_n=_weighted_mean(n_arr, w),
        frac_singleton_stacks=n1 / max(1.0, float(n_stacks)),
        d_char_n2_km=d_char_mean,
        dens_per_km2=dens,
    )


def build_radius_profile(clusters: list[Cluster]) -> RadiusProfile:
    """Return a |lat| interpolator for R, D, and cluster density.

    Args:
        clusters: World clusters; ISS-belt filter applied here.

    Returns:
        RadiusProfile over 2 deg |lat| bins from 0 to inclination.
    """
    iss = [c for c in clusters if abs(c.lat) <= TLE.inclination_deg]
    edges = np.arange(0.0, TLE.inclination_deg + LAT_BIN_DEG, LAT_BIN_DEG)
    if float(edges[-1]) < TLE.inclination_deg:
        edges = np.append(edges, TLE.inclination_deg)
    rows: list[LatBandRow] = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        closed = i == edges.size - 2
        rows.append(_band_row(iss, float(lo), float(hi), closed_hi=closed))
    abs_lat = np.array([row.lat_mid for row in rows], dtype=float)
    mean_r = np.array([row.mean_r_km for row in rows], dtype=float)
    mean_d = np.array([row.mean_d_km for row in rows], dtype=float)
    dens = np.array([row.dens_per_km2 for row in rows], dtype=float)
    if rows and rows[0].n_stacks == 0:
        mean_r[0] = PLUME_R_KM
    for i in range(1, mean_r.size):
        if rows[i].n_stacks == 0:
            mean_r[i] = mean_r[i - 1]
            mean_d[i] = mean_d[i - 1]
            dens[i] = dens[i - 1]
    return RadiusProfile(
        abs_lat=abs_lat,
        mean_r_km=mean_r,
        mean_d_km=mean_d,
        dens_per_km2=dens,
        rows=tuple(rows),
    )


def folded_band_rows(clusters: list[Cluster]) -> list[LatBandRow]:
    """Return stats for the coarse folded |lat| reporting bands.

    Args:
        clusters: World clusters; ISS-belt filter applied here.

    Returns:
        One row per FOLDED_LAT_BANDS entry.
    """
    iss = [c for c in clusters if abs(c.lat) <= TLE.inclination_deg]
    out: list[LatBandRow] = []
    for i, (lo, hi) in enumerate(FOLDED_LAT_BANDS):
        closed = i == len(FOLDED_LAT_BANDS) - 1
        out.append(_band_row(iss, lo, hi, closed_hi=closed))
    return out


def ground_speed_km_s(orbit: Orbit, lat_deg: float) -> float:
    """Return ISS ground-track speed at ``lat_deg``.

    Circular inertial speed scaled by Earth radius over ISS radius.

    Args:
        orbit: Circular ISS orbit.
        lat_deg: Geocentric latitude in degrees.

    Returns:
        Ground-track speed in km/s.
    """
    re_km = orbit.earth_radius_km(lat_deg)
    return orbit.v_km_s * re_km / orbit.radius_km


def encounter_time_s(dens_per_km2: float, swath_km: float, v_scan_km_s: float) -> float:
    """Return mean time to the next cluster along a scanned swath.

    Args:
        dens_per_km2: Cluster density per square kilometre.
        swath_km: Cross-track search width in kilometres.
        v_scan_km_s: Along-track scan speed in km/s.

    Returns:
        Mean encounter time in seconds, or inf if density or speed is 0.
    """
    rate = dens_per_km2 * swath_km * v_scan_km_s
    if rate <= 0.0:
        return math.inf
    return 1.0 / rate


def reacquire_s(t_reset_s: float, t_enc_s: float) -> float:
    """Return mean time from loss of one target to acquire of the next.

    After a target leaves the frame the gimbal slews to the elevation-window
    start, then waits for the next cluster to enter the search swath.

    Args:
        t_reset_s: Slew time back to the window start, seconds.
        t_enc_s: Mean along-track encounter time in seconds.

    Returns:
        t_reset_s + t_enc_s, or inf if either is not finite.
    """
    if not math.isfinite(t_reset_s) or not math.isfinite(t_enc_s):
        return math.inf
    if t_reset_s < 0.0 or t_enc_s < 0.0:
        return math.inf
    return t_reset_s + t_enc_s


def cycle_s(t_track_s: float, t_reacq_s: float) -> float:
    """Return time from start of tracking to reacquire of the next target.

    Args:
        t_track_s: Single-target in-frame dwell in seconds.
        t_reacq_s: Mean reacquire time in seconds.

    Returns:
        t_track_s + t_reacq_s, or inf if reacquire is not finite.
    """
    if not math.isfinite(t_reacq_s):
        return math.inf
    return t_track_s + t_reacq_s


def reset_to_start_s() -> float:
    """Return hardware slew time from the 30 deg stop back to nadir start.

    Returns:
        Reset seconds at MAX_HW_SLEW_DEG_S.
    """
    span = GIMBAL_BOX.el_nadir_deg - GIMBAL_BOX.el_limb_deg
    return span / MAX_HW_SLEW_DEG_S


def hunt_at_lat(
    lat_deg: float,
    dens_per_km2: float,
    optics: Optics,
    orbit: Orbit,
) -> tuple[float, float, float, float, float]:
    """Return reset and reacquire times at one latitude.

    After loss the gimbal slews to the elevation-window start (hardware
    cap). Mean wait for the next cluster uses ISS ground-track speed
    through the 1-axis FOV ribbon or the 2-axis gimbal box.

    Args:
        lat_deg: Geocentric latitude in degrees.
        dens_per_km2: Cluster density per square kilometre.
        optics: Usable sensor FOV.
        orbit: Circular ISS orbit.

    Returns:
        (t_reset, t_enc_1axis, t_enc_2axis, t_reacq_1axis, t_reacq_2axis).
    """
    t_reset = reset_to_start_s()
    h_km = orbit.local_altitude_km(abs(lat_deg))
    swath1 = 2.0 * h_km * math.tan(math.radians(optics.half_az_deg))
    swath2 = 2.0 * h_km * math.tan(math.radians(GIMBAL_BOX.az_box_deg))
    v_g = ground_speed_km_s(orbit, lat_deg)
    t_enc1 = encounter_time_s(dens_per_km2, swath1, v_g)
    t_enc2 = encounter_time_s(dens_per_km2, swath2, v_g)
    return t_reset, t_enc1, t_enc2, reacquire_s(t_reset, t_enc1), reacquire_s(t_reset, t_enc2)
