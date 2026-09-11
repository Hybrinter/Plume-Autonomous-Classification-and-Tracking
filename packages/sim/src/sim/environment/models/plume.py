"""Plume models: CI band-plane Gaussian and a frozen ECEF column."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# third-party
import numpy as np

# internal
from flight.hal.interfaces.ephemeris import IssState
from flight.payload.gimbal.geo import ecef_from_eci

from sim.environment.models.earth import EarthModel
from sim.environment.models.orbit import OrbitModel
from sim.environment.models.wind import WindModel
from sim.environment.records import EnvTime, PlumeState

# Band-plane centroid of sim.scene.plume (documented, not a live import of privates).
_BANDPLANE_X_PX = 612.0
_BANDPLANE_Y_PX = 124.0


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
    ) -> PlumeState:
        """Return plume state. Band-plane models ignore ISS and wind."""
        ...


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
    ) -> PlumeState:
        """Return a band-plane CoG at the CI plume centroid."""
        del time, prior, wind, earth, iss, orbit, omega_earth_rad_s, epoch_utc_s
        return PlumeState(
            frame="bandplane",
            cog_ecef_m=None,
            centroid_band_px=(_BANDPLANE_X_PX, _BANDPLANE_Y_PX),
            along_sigma_m=self.along_sigma_m,
            cross_sigma_m=self.cross_sigma_m,
            height_proxy_m=self.height_proxy_m,
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
    ) -> PlumeState:
        """Return a frozen ECEF CoG. Wind is inert until advected_gaussian."""
        del time, prior, wind, orbit
        cog = self.cog_ecef_m
        if cog is None:
            r_iss = np.asarray(iss.r_m, dtype=np.float64)
            r_ecef = ecef_from_eci(r_iss, omega_earth_rad_s, iss.epoch_utc_s, epoch_utc_s)
            nadir = -r_ecef / float(np.linalg.norm(r_ecef))
            hit = earth.intersect_at_height(r_ecef, nadir, self.height_proxy_m)
            if hit is None:
                return PlumeState(
                    frame="ecef",
                    cog_ecef_m=None,
                    centroid_band_px=None,
                    along_sigma_m=self.along_sigma_m,
                    cross_sigma_m=self.cross_sigma_m,
                    height_proxy_m=self.height_proxy_m,
                )
            point, _slant = hit
            cog = (float(point[0]), float(point[1]), float(point[2]))
        return PlumeState(
            frame="ecef",
            cog_ecef_m=cog,
            centroid_band_px=None,
            along_sigma_m=self.along_sigma_m,
            cross_sigma_m=self.cross_sigma_m,
            height_proxy_m=self.height_proxy_m,
        )
