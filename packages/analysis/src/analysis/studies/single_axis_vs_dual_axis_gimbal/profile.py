"""Latitude-binned plant span, covering radius, and hunt encounter rates.

Stack-weighted D(|lat|) comes from Climate TRACE clusters. Covering radius
is R = D + PLUME_R_KM. After a target leaves the frame the leftover-window
rewind is already gated so a new target just outside the frame can be
acquired immediately -- there is no wait to reset to 30 deg. Mean reacquire
is the encounter wait from signed-latitude stack density. Cycle time is
single-target dwell plus reacquire.

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
        dens_per_km2: Stack count per square kilometre of the folded band.
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
    """Interpolators for R(|lat|) and signed-latitude stack density.

    Attributes:
        abs_lat: Bin midpoints in degrees, increasing. # np.ndarray[float64, (B,)]
        mean_r_km: Stack-weighted mean R at those midpoints.
        mean_d_km: Stack-weighted mean D at those midpoints.
        rows: One LatBandRow per |lat| bin, including empty bins.
        signed_lat: Signed-latitude bin midpoints, increasing.
        stack_dens_per_km2: Stack area density at those signed midpoints.
    """

    abs_lat: np.ndarray
    mean_r_km: np.ndarray
    mean_d_km: np.ndarray
    rows: tuple[LatBandRow, ...]
    signed_lat: np.ndarray
    stack_dens_per_km2: np.ndarray

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
        """Return interpolated stack density at signed ``lat_deg``.

        Northern and southern bands are distinct. Density follows the
        industrial inventory (peak near 30-40 N), not a folded |lat| mean.

        Args:
            lat_deg: Geocentric latitude in degrees (signed).

        Returns:
            Stacks per square kilometre.
        """
        lo = float(self.signed_lat[0])
        hi = float(self.signed_lat[-1])
        x = min(hi, max(lo, float(lat_deg)))
        return float(np.interp(x, self.signed_lat, self.stack_dens_per_km2))


def _folded_band_area_km2(lat_lo: float, lat_hi: float) -> float:
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


def _signed_band_area_km2(lat_lo: float, lat_hi: float) -> float:
    """Return spherical area of one signed-latitude strip, km2.

    Args:
        lat_lo: Low signed-latitude edge in degrees.
        lat_hi: High signed-latitude edge in degrees.

    Returns:
        Area in square kilometres. Zero if the edges are equal.
    """
    lo = math.radians(lat_lo)
    hi = math.radians(lat_hi)
    if hi == lo:
        return 0.0
    radius = MEAN_EARTH_RADIUS_KM
    return 2.0 * math.pi * radius * radius * abs(math.sin(hi) - math.sin(lo))


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
    area = _folded_band_area_km2(lat_lo, lat_hi)
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
    dens = float(n_stacks / area) if area > 0.0 else 0.0
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


def _signed_stack_density(iss: list[Cluster]) -> tuple[np.ndarray, np.ndarray]:
    """Return signed-latitude midpoints and stack area density.

    Each 2 deg band uses that hemisphere's stack count over that strip's
    spherical area. Empty bands stay at density 0.

    Args:
        iss: ISS-belt clusters.

    Returns:
        (signed_lat_mid_deg, stack_dens_per_km2).
    """
    n_bins = int(math.ceil(TLE.inclination_deg / LAT_BIN_DEG))
    edges = np.arange(
        -n_bins * LAT_BIN_DEG,
        (n_bins + 1) * LAT_BIN_DEG,
        LAT_BIN_DEG,
        dtype=float,
    )
    mids = 0.5 * (edges[:-1] + edges[1:])
    dens = np.zeros(mids.size)
    if not iss:
        return mids, dens
    lats = np.array([c.lat for c in iss], dtype=float)
    stacks = np.array([float(c.n) for c in iss], dtype=float)
    last = mids.size - 1
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        if i == last:
            mask = (lats >= lo) & (lats <= hi)
        else:
            mask = (lats >= lo) & (lats < hi)
        n_stacks = float(np.sum(stacks[mask]))
        area = _signed_band_area_km2(float(lo), float(hi))
        dens[i] = n_stacks / area if area > 0.0 else 0.0
    return mids, dens


def build_radius_profile(clusters: list[Cluster]) -> RadiusProfile:
    """Return interpolators for R(|lat|) and signed-latitude stack density.

    Args:
        clusters: World clusters; ISS-belt filter applied here.

    Returns:
        RadiusProfile over 2 deg |lat| bins and signed 2 deg stack-density bins.
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
    if rows and rows[0].n_stacks == 0:
        mean_r[0] = PLUME_R_KM
    for i in range(1, mean_r.size):
        if rows[i].n_stacks == 0:
            mean_r[i] = mean_r[i - 1]
            mean_d[i] = mean_d[i - 1]
    signed_lat, stack_dens = _signed_stack_density(iss)
    return RadiusProfile(
        abs_lat=abs_lat,
        mean_r_km=mean_r,
        mean_d_km=mean_d,
        rows=tuple(rows),
        signed_lat=signed_lat,
        stack_dens_per_km2=stack_dens,
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
    """Return mean time to the next stack along a scanned swath.

    Args:
        dens_per_km2: Stack density per square kilometre.
        swath_km: Cross-track search width in kilometres.
        v_scan_km_s: Along-track scan speed in km/s.

    Returns:
        Mean encounter time in seconds, or inf if density or speed is 0.
    """
    rate = dens_per_km2 * swath_km * v_scan_km_s
    if rate <= 0.0:
        return math.inf
    return 1.0 / rate


def reacquire_s(t_enc_s: float) -> float:
    """Return mean time from loss of one target to acquire of the next.

    Leftover-window rewind is gated so a target just outside the frame can
    be acquired immediately. There is no wait to reset to 30 deg. Lost time
    is the mean encounter wait at this signed latitude.

    Args:
        t_enc_s: Mean along-track encounter time in seconds.

    Returns:
        t_enc_s, or inf if it is not finite and non-negative.
    """
    if not math.isfinite(t_enc_s) or t_enc_s < 0.0:
        return math.inf
    return t_enc_s


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


def hunt_at_lat(
    lat_deg: float,
    dens_per_km2: float,
    optics: Optics,
    orbit: Orbit,
) -> tuple[float, float]:
    """Return 1-axis and 2-axis mean reacquire times at signed latitude.

    No reset-to-30-deg wait: a new target just outside the frame can be
    acquired immediately. Mean wait uses ISS ground-track speed through
    the 1-axis FOV ribbon or the 2-axis gimbal box, times stack density
    at this signed latitude.

    Args:
        lat_deg: Geocentric latitude in degrees (signed).
        dens_per_km2: Stack density per square kilometre at this latitude.
        optics: Usable sensor FOV.
        orbit: Circular ISS orbit.

    Returns:
        (t_reacq_1axis, t_reacq_2axis).
    """
    h_km = orbit.local_altitude_km(abs(lat_deg))
    swath1 = 2.0 * h_km * math.tan(math.radians(optics.half_az_deg))
    swath2 = 2.0 * h_km * math.tan(math.radians(GIMBAL_BOX.az_box_deg))
    v_g = ground_speed_km_s(orbit, lat_deg)
    t_enc1 = encounter_time_s(dens_per_km2, swath1, v_g)
    t_enc2 = encounter_time_s(dens_per_km2, swath2, v_g)
    return reacquire_s(t_enc1), reacquire_s(t_enc2)
