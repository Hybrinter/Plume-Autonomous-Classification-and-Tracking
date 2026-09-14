"""Plume models: CI band-plane Gaussian, frozen ECEF column, latitude Poisson.

PoissonLatitude draws the next along-track CoG from a 1-D Poisson process in
a ground-track corridor. Intensity is dens(lat) times twice the cross-track
half-width. FOV and hunt waits are observer geometry, not this model.

Contains:
  - PlumeModel, BandplaneGaussian, EcefColumn, PoissonLatitude
  - along_track_intensity_per_km
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

# third-party
import numpy as np

# internal
from flight.hal.interfaces.ephemeris import IssState
from flight.payload.gimbal.geo import ecef_from_eci, eci_from_ecef, lvlh_axes

from sim.environment.models.earth import EarthModel
from sim.environment.models.orbit import OrbitModel
from sim.environment.models.wind import WindModel
from sim.environment.records import EnvTime, PlumeState

# Band-plane centroid of sim.scene.plume (documented, not a live import of privates).
_BANDPLANE_X_PX = 612.0
_BANDPLANE_Y_PX = 124.0
_MIN_NORM = 1.0e-12
_PASSED_AHEAD_M = 1.0


@runtime_checkable
class PlumeModel(Protocol):
    """Plume state at a shutter. One evaluate signature for every named model."""

    def evaluate(
        self,
        time: EnvTime,
        prior: PlumeState | None,
        wind: WindModel,
        earth: EarthModel,
        iss: IssState,
        orbit: OrbitModel,
        omega_earth_rad_s: float,
        epoch_utc_s: float,
        rng: np.random.Generator,
    ) -> PlumeState:
        """Return plume state. Band-plane models ignore ISS, wind, and rng."""
        ...


def along_track_intensity_per_km(dens_per_km2: float, cross_track_half_km: float) -> float:
    """Return 1-D Poisson intensity along the ground track, stacks per kilometre.

    Args:
        dens_per_km2: Area density of stacks.
        cross_track_half_km: Half-width of the generation corridor.

    Returns:
        dens_per_km2 * 2 * cross_track_half_km. Zero when either factor is not
        positive.
    """
    if dens_per_km2 <= 0.0 or cross_track_half_km <= 0.0:
        return 0.0
    return dens_per_km2 * 2.0 * cross_track_half_km


def _ecef_state(
    frame: Literal["ecef", "bandplane"],
    cog_ecef_m: tuple[float, float, float] | None,
    along_sigma_m: float,
    cross_sigma_m: float,
    height_proxy_m: float,
    *,
    present: bool,
    centroid_band_px: tuple[float, float] | None = None,
) -> PlumeState:
    """Build a PlumeState with an explicit present flag."""
    return PlumeState(
        frame=frame,
        cog_ecef_m=cog_ecef_m,
        centroid_band_px=centroid_band_px,
        along_sigma_m=along_sigma_m,
        cross_sigma_m=cross_sigma_m,
        height_proxy_m=height_proxy_m,
        present=present,
    )


def _nadir_hit_ecef(
    earth: EarthModel,
    iss: IssState,
    height_proxy_m: float,
    omega_earth_rad_s: float,
    epoch_utc_s: float,
) -> tuple[float, float, float] | None:
    """Return the nadir height-proxy hit in ECEF, or None."""
    r_iss = np.asarray(iss.r_m, dtype=np.float64)
    r_ecef = ecef_from_eci(r_iss, omega_earth_rad_s, iss.epoch_utc_s, epoch_utc_s)
    nadir = -r_ecef / float(np.linalg.norm(r_ecef))
    hit = earth.intersect_at_height(r_ecef, nadir, height_proxy_m)
    if hit is None:
        return None
    point, _slant = hit
    return (float(point[0]), float(point[1]), float(point[2]))


def _geocentric_lat_deg(r_ecef_m: np.ndarray) -> float:
    """Return geocentric latitude in degrees from an ECEF vector."""
    return math.degrees(
        math.atan2(float(r_ecef_m[2]), math.hypot(float(r_ecef_m[0]), float(r_ecef_m[1])))
    )


def _along_track_ahead_m(
    iss: IssState,
    cog_ecef_m: tuple[float, float, float],
    omega_earth_rad_s: float,
    epoch_utc_s: float,
) -> float:
    """Return LVLH along-track metres of CoG minus ISS (positive ahead)."""
    r_iss = np.asarray(iss.r_m, dtype=np.float64)
    v_iss = np.asarray(iss.v_m_s, dtype=np.float64)
    cog_eci = eci_from_ecef(
        np.asarray(cog_ecef_m, dtype=np.float64),
        omega_earth_rad_s,
        iss.epoch_utc_s,
        epoch_utc_s,
    )
    x_hat, _y_hat, _z_hat = lvlh_axes(r_iss, v_iss)
    return float(np.dot(cog_eci - r_iss, x_hat))


@dataclass(frozen=True, slots=True)
class BandplaneGaussian:
    """Fixed band-plane blob. Ignores Earth, orbit, wind, and shutter elevation."""

    height_proxy_m: float = 2000.0
    along_sigma_m: float = 0.0
    cross_sigma_m: float = 0.0

    def evaluate(
        self,
        time: EnvTime,
        prior: PlumeState | None,
        wind: WindModel,
        earth: EarthModel,
        iss: IssState,
        orbit: OrbitModel,
        omega_earth_rad_s: float,
        epoch_utc_s: float,
        rng: np.random.Generator,
    ) -> PlumeState:
        """Return a band-plane CoG at the CI plume centroid."""
        del time, prior, wind, earth, iss, orbit, omega_earth_rad_s, epoch_utc_s, rng
        return _ecef_state(
            "bandplane",
            None,
            self.along_sigma_m,
            self.cross_sigma_m,
            self.height_proxy_m,
            present=True,
            centroid_band_px=(_BANDPLANE_X_PX, _BANDPLANE_Y_PX),
        )


@dataclass(frozen=True, slots=True)
class EcefColumn:
    """Gaussian column on a frozen ECEF CoG at the height proxy.

    If cog_ecef_m is None, place the CoG at the nadir height-proxy hit at this
    ISS state (first closed-loop default).
    """

    cog_ecef_m: tuple[float, float, float] | None = None
    along_sigma_m: float = 200.0
    cross_sigma_m: float = 200.0
    height_proxy_m: float = 2000.0

    def evaluate(
        self,
        time: EnvTime,
        prior: PlumeState | None,
        wind: WindModel,
        earth: EarthModel,
        iss: IssState,
        orbit: OrbitModel,
        omega_earth_rad_s: float,
        epoch_utc_s: float,
        rng: np.random.Generator,
    ) -> PlumeState:
        """Return a frozen ECEF CoG. Wind is inert until advected_gaussian."""
        del time, prior, wind, orbit, rng
        cog = self.cog_ecef_m
        if cog is None:
            cog = _nadir_hit_ecef(earth, iss, self.height_proxy_m, omega_earth_rad_s, epoch_utc_s)
            if cog is None:
                return _ecef_state(
                    "ecef",
                    None,
                    self.along_sigma_m,
                    self.cross_sigma_m,
                    self.height_proxy_m,
                    present=False,
                )
        return _ecef_state(
            "ecef",
            cog,
            self.along_sigma_m,
            self.cross_sigma_m,
            self.height_proxy_m,
            present=True,
        )


@dataclass(frozen=True, slots=True)
class PoissonLatitude:
    """Along-track Poisson stacks at signed-latitude area density.

    Tables are piecewise-linear in geocentric latitude (degrees). Intensity
    along-track is dens(lat) times the corridor width 2 * cross_track_half_km.
    The next CoG lives in prior until the ISS along-track coordinate passes it.
    """

    signed_lat_deg: tuple[float, ...]
    dens_per_km2: tuple[float, ...]
    along_sigma_m: float = 200.0
    cross_sigma_m: float = 200.0
    height_proxy_m: float = 2000.0
    cross_track_half_km: float = 5.0

    def density_per_km2(self, lat_deg: float) -> float:
        """Interpolate stack area density at geocentric latitude.

        Args:
            lat_deg: Geocentric latitude in degrees (signed).

        Returns:
            dens_per_km2 from the table. Endpoints hold outside the span.
        """
        lat = np.asarray(self.signed_lat_deg, dtype=np.float64)
        dens = np.asarray(self.dens_per_km2, dtype=np.float64)
        return float(np.interp(lat_deg, lat, dens))

    def evaluate(
        self,
        time: EnvTime,
        prior: PlumeState | None,
        wind: WindModel,
        earth: EarthModel,
        iss: IssState,
        orbit: OrbitModel,
        omega_earth_rad_s: float,
        epoch_utc_s: float,
        rng: np.random.Generator,
    ) -> PlumeState:
        """Return the active along-track CoG, or absent when density is zero."""
        del time, wind
        if (
            prior is not None
            and prior.present
            and prior.cog_ecef_m is not None
            and _along_track_ahead_m(iss, prior.cog_ecef_m, omega_earth_rad_s, epoch_utc_s)
            > _PASSED_AHEAD_M
        ):
            return prior
        ssp = _nadir_hit_ecef(earth, iss, self.height_proxy_m, omega_earth_rad_s, epoch_utc_s)
        if ssp is None:
            return _ecef_state(
                "ecef",
                None,
                self.along_sigma_m,
                self.cross_sigma_m,
                self.height_proxy_m,
                present=False,
            )
        lat_deg = _geocentric_lat_deg(np.asarray(ssp, dtype=np.float64))
        dens = self.density_per_km2(lat_deg)
        lam = along_track_intensity_per_km(dens, self.cross_track_half_km)
        if lam <= 0.0:
            return _ecef_state(
                "ecef",
                None,
                self.along_sigma_m,
                self.cross_sigma_m,
                self.height_proxy_m,
                present=False,
            )
        ds_km = float(rng.exponential(1.0 / lam))
        cross_km = float(rng.uniform(-self.cross_track_half_km, self.cross_track_half_km))
        cog = _place_along_track(
            earth,
            orbit,
            iss,
            ds_km * 1000.0,
            cross_km * 1000.0,
            self.height_proxy_m,
            omega_earth_rad_s,
            epoch_utc_s,
        )
        if cog is None:
            return _ecef_state(
                "ecef",
                None,
                self.along_sigma_m,
                self.cross_sigma_m,
                self.height_proxy_m,
                present=False,
            )
        return _ecef_state(
            "ecef",
            cog,
            self.along_sigma_m,
            self.cross_sigma_m,
            self.height_proxy_m,
            present=True,
        )


def _ssp_ground_separation_m(
    ssp_a_ecef_m: np.ndarray,
    ssp_b_ecef_m: np.ndarray,
) -> float:
    """Return arc length between successive height-proxy nadir hits."""
    ra = float(np.linalg.norm(ssp_a_ecef_m))
    rb = float(np.linalg.norm(ssp_b_ecef_m))
    if ra < _MIN_NORM or rb < _MIN_NORM:
        return 0.0
    cos_angle = float(np.dot(ssp_a_ecef_m, ssp_b_ecef_m) / (ra * rb))
    cos_angle = min(1.0, max(-1.0, cos_angle))
    angle = math.acos(cos_angle)
    return 0.5 * (ra + rb) * angle


def _advance_iss_along_ground_track(
    earth: EarthModel,
    orbit: OrbitModel,
    iss: IssState,
    along_m: float,
    height_proxy_m: float,
    omega_earth_rad_s: float,
    epoch_utc_s: float,
) -> IssState | None:
    """Step orbit forward until cumulative nadir ground-track distance reaches along_m."""
    if along_m <= 0.0:
        return iss
    ssp = _nadir_hit_ecef(earth, iss, height_proxy_m, omega_earth_rad_s, epoch_utc_s)
    if ssp is None:
        return None
    ssp_ecef = np.asarray(ssp, dtype=np.float64)
    t_s = iss.epoch_utc_s
    dist_m = 0.0
    dt_s = 1.0
    while dist_m < along_m:
        t_next_s = t_s + dt_s
        iss_next = orbit.state_eci(t_next_s)
        ssp_next = _nadir_hit_ecef(earth, iss_next, height_proxy_m, omega_earth_rad_s, epoch_utc_s)
        if ssp_next is None:
            return None
        ssp_next_ecef = np.asarray(ssp_next, dtype=np.float64)
        seg_m = _ssp_ground_separation_m(ssp_ecef, ssp_next_ecef)
        if seg_m < _MIN_NORM:
            t_s = t_next_s
            iss = iss_next
            ssp_ecef = ssp_next_ecef
            continue
        if dist_m + seg_m >= along_m:
            frac = (along_m - dist_m) / seg_m
            return orbit.state_eci(t_s + frac * dt_s)
        dist_m += seg_m
        t_s = t_next_s
        iss = iss_next
        ssp_ecef = ssp_next_ecef
    return iss


def _cog_at_cross_offset(
    earth: EarthModel,
    iss: IssState,
    cross_m: float,
    height_proxy_m: float,
    omega_earth_rad_s: float,
    epoch_utc_s: float,
) -> tuple[float, float, float] | None:
    """Place CoG at the height-proxy hit after a LVLH starboard offset from nadir."""
    ssp = _nadir_hit_ecef(earth, iss, height_proxy_m, omega_earth_rad_s, epoch_utc_s)
    if ssp is None:
        return None
    r_iss = np.asarray(iss.r_m, dtype=np.float64)
    v_iss = np.asarray(iss.v_m_s, dtype=np.float64)
    _x_hat, y_hat, _z_hat = lvlh_axes(r_iss, v_iss)
    ssp_eci = eci_from_ecef(
        np.asarray(ssp, dtype=np.float64),
        omega_earth_rad_s,
        iss.epoch_utc_s,
        epoch_utc_s,
    )
    anchor_eci = ssp_eci + y_hat * cross_m
    r_iss_ecef = ecef_from_eci(r_iss, omega_earth_rad_s, iss.epoch_utc_s, epoch_utc_s)
    anchor_ecef = ecef_from_eci(anchor_eci, omega_earth_rad_s, iss.epoch_utc_s, epoch_utc_s)
    look = anchor_ecef - r_iss_ecef
    look_n = float(np.linalg.norm(look))
    if look_n < _MIN_NORM:
        return ssp
    hit = earth.intersect_at_height(r_iss_ecef, look / look_n, height_proxy_m)
    if hit is None:
        return None
    point, _slant = hit
    return (float(point[0]), float(point[1]), float(point[2]))


def _place_along_track(
    earth: EarthModel,
    orbit: OrbitModel,
    iss: IssState,
    along_m: float,
    cross_m: float,
    height_proxy_m: float,
    omega_earth_rad_s: float,
    epoch_utc_s: float,
) -> tuple[float, float, float] | None:
    """Advance along ground track, then apply a cross-track height-proxy intersect."""
    iss_ahead = _advance_iss_along_ground_track(
        earth,
        orbit,
        iss,
        along_m,
        height_proxy_m,
        omega_earth_rad_s,
        epoch_utc_s,
    )
    if iss_ahead is None:
        return None
    return _cog_at_cross_offset(
        earth,
        iss_ahead,
        cross_m,
        height_proxy_m,
        omega_earth_rad_s,
        epoch_utc_s,
    )
