"""Circular ISS orbit from a two-line mean-motion TLE, plus ECI kinematics.

Thin km/deg wrappers over sim ``CircularKepler`` and
``geocentric_radius_m``. SI stays inside sim; this module returns km/deg
at the study boundary.

Contains:
  - IssTle: mean elements used to build the orbit.
  - Orbit: SMA, radius, rates, and local altitude vs latitude.
  - wgs84_geocentric_radius_km: WGS-84 geocentric radius.
  - ephemeris_from_tle / build_orbit / argument_of_latitude / heading_from_north_deg.
  - iss_eci / origin_ecef / offset_ecef.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
from flight.libs.config import EphemerisConfig
from sim.environment.models.earth import geocentric_radius_m
from sim.environment.models.orbit import CircularKepler

from analysis.studies.single_axis_vs_dual_axis_gimbal.constants import MU_KM3_S2, WGS84_A_KM


@dataclass(frozen=True)
class IssTle:
    """Mean Keplerian elements from a GP TLE (circular-orbit subset).

    Attributes:
        epoch: TLE epoch date string (documentation only).
        inclination_deg: Inclination in degrees.
        eccentricity: Eccentricity (used for perigee/apogee altitudes).
        mean_motion_rev_per_day: Mean motion in revolutions per day.
    """

    epoch: str
    inclination_deg: float
    eccentricity: float
    mean_motion_rev_per_day: float


@dataclass(frozen=True)
class Orbit:
    """Circular ISS orbit used for look-angle sampling.

    Attributes:
        sma_km: Semi-major axis from mean motion.
        radius_km: Geocentric radius used for this run (SMA or perigee).
        inclination_rad: Inclination in radians.
        eccentricity: TLE eccentricity.
        n_rad_s: Mean motion for the chosen radius, rad/s.
        v_km_s: Inertial speed at the chosen radius.
        period_s: Orbit period at the chosen radius.
        perigee_alt_eq_km: Perigee altitude above the WGS-84 equator.
        apogee_alt_eq_km: Apogee altitude above the WGS-84 equator.
        mean_alt_eq_km: SMA minus WGS-84 equatorial radius.
    """

    sma_km: float
    radius_km: float
    inclination_rad: float
    eccentricity: float
    n_rad_s: float
    v_km_s: float
    period_s: float
    perigee_alt_eq_km: float
    apogee_alt_eq_km: float
    mean_alt_eq_km: float

    @property
    def inclination_deg(self) -> float:
        """Return inclination in degrees."""
        return math.degrees(self.inclination_rad)

    def local_altitude_km(self, lat_deg: float) -> float:
        """Return ISS altitude above the WGS-84 geocentric radius at ``lat_deg``.

        Args:
            lat_deg: Geocentric latitude in degrees.

        Returns:
            Altitude in kilometres.
        """
        return self.radius_km - wgs84_geocentric_radius_km(math.radians(lat_deg))

    def ground_speed_km_s(self, lat_deg: float) -> float:
        """Return ISS ground-track speed at ``lat_deg``.

        Circular inertial speed scaled by Earth radius over ISS radius.

        Args:
            lat_deg: Geocentric latitude in degrees.

        Returns:
            Ground-track speed in kilometres per second.
        """
        re_km = self.earth_radius_km(lat_deg)
        return self.v_km_s * re_km / self.radius_km

    def earth_radius_km(self, lat_deg: float) -> float:
        """Return WGS-84 geocentric Earth radius at ``lat_deg``.

        Args:
            lat_deg: Geocentric latitude in degrees.

        Returns:
            Geocentric radius in kilometres.
        """
        return wgs84_geocentric_radius_km(math.radians(lat_deg))


def wgs84_geocentric_radius_km(lat_rad: float) -> float:
    """Return WGS-84 geocentric radius at geocentric latitude ``lat_rad``.

    Args:
        lat_rad: Geocentric latitude in radians.

    Returns:
        Radius in kilometres from sim ``geocentric_radius_m``.
    """
    cfg = EphemerisConfig()
    return geocentric_radius_m(cfg.wgs84_a_m, cfg.wgs84_f, lat_rad) / 1000.0


def ephemeris_from_tle(tle: IssTle) -> EphemerisConfig:
    """Return flight ephemeris constants with this TLE's mean elements.

    Args:
        tle: Mean Keplerian elements.

    Returns:
        EphemerisConfig. WGS-84 and mu stay the flight defaults.
    """
    base = EphemerisConfig()
    return EphemerisConfig(
        inclination_deg=tle.inclination_deg,
        mean_motion_rev_per_day=tle.mean_motion_rev_per_day,
        mu_m3_s2=base.mu_m3_s2,
        omega_earth_rad_s=base.omega_earth_rad_s,
        wgs84_a_m=base.wgs84_a_m,
        wgs84_f=base.wgs84_f,
        epoch_utc_s=base.epoch_utc_s,
    )


def build_orbit(tle: IssTle, *, use_perigee: bool = False) -> Orbit:
    """Build a circular orbit from TLE mean motion.

    Args:
        tle: Mean elements.
        use_perigee: If True, use perigee radius instead of SMA.

    Returns:
        Orbit at SMA (default) or perigee.
    """
    n_mean = tle.mean_motion_rev_per_day * 2.0 * math.pi / 86400.0
    sma = (MU_KM3_S2 / n_mean**2) ** (1.0 / 3.0)
    radius_perigee = sma * (1.0 - tle.eccentricity)
    radius_apogee = sma * (1.0 + tle.eccentricity)
    radius = radius_perigee if use_perigee else sma
    speed = math.sqrt(MU_KM3_S2 / radius)
    n_used = math.sqrt(MU_KM3_S2 / radius**3)
    return Orbit(
        sma_km=sma,
        radius_km=radius,
        inclination_rad=math.radians(tle.inclination_deg),
        eccentricity=tle.eccentricity,
        n_rad_s=n_used,
        v_km_s=speed,
        period_s=2.0 * math.pi / n_used,
        perigee_alt_eq_km=radius_perigee - WGS84_A_KM,
        apogee_alt_eq_km=radius_apogee - WGS84_A_KM,
        mean_alt_eq_km=sma - WGS84_A_KM,
    )


def argument_of_latitude(lat_deg: float, inclination_rad: float) -> float:
    """Return northern-hemisphere argument of latitude for ``lat_deg``.

    Args:
        lat_deg: Geocentric latitude in degrees.
        inclination_rad: Orbit inclination in radians.

    Returns:
        Argument of latitude in radians, clipped to the inclination.
    """
    s_val = math.sin(math.radians(lat_deg)) / math.sin(inclination_rad)
    s_val = max(-1.0, min(1.0, s_val))
    return math.asin(s_val)


def heading_from_north_deg(lat_deg: float, inclination_rad: float) -> float:
    """Return ground-track azimuth from north at ``lat_deg`` (ascending).

    Args:
        lat_deg: Geocentric latitude in degrees.
        inclination_rad: Orbit inclination in radians.

    Returns:
        Heading in degrees from north toward east.
    """
    cphi = math.cos(math.radians(lat_deg))
    if abs(cphi) < 1e-12:
        return 90.0
    s_val = min(1.0, math.cos(inclination_rad) / cphi)
    return math.degrees(math.asin(s_val))


def iss_eci(t_s: float, orbit: Orbit, u0: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ISS ECI position and velocity for a circular orbit.

    Wraps sim ``CircularKepler`` (meters) and converts to kilometres.
    Node is on the X axis. ``u0`` is argument of latitude at t = 0.

    Args:
        t_s: Seconds from the t = 0 sub-satellite epoch.
        orbit: Circular orbit.
        u0: Argument of latitude at t = 0, radians.

    Returns:
        Position (km) and inertial velocity (km/s). # np.ndarray[float64, (3,)]
    """
    mean_motion = orbit.n_rad_s * 86400.0 / (2.0 * math.pi)
    kepler = CircularKepler(
        inclination_deg=orbit.inclination_deg,
        mean_motion_rev_per_day=mean_motion,
        mu_m3_s2=MU_KM3_S2 * 1.0e9,
        epoch_utc_s=0.0,
    )
    utc_s = float(t_s) + u0 / orbit.n_rad_s
    state = kepler.state_eci(utc_s)
    pos = np.asarray(state.r_m, dtype=np.float64) / 1000.0
    vel = np.asarray(state.v_m_s, dtype=np.float64) / 1000.0
    return pos, vel


def origin_ecef(orbit: Orbit, lat_deg: float) -> np.ndarray:
    """Return the cluster CoG on the Earth surface at the t = 0 sub-satellite point.

    ECI and ECEF are aligned at t = 0.

    Args:
        orbit: Circular orbit.
        lat_deg: Geocentric latitude of the pass in degrees.

    Returns:
        ECEF position in kilometres. # np.ndarray[float64, (3,)]
    """
    u0 = argument_of_latitude(lat_deg, orbit.inclination_rad)
    r0, _ = iss_eci(0.0, orbit, u0)
    earth_r = orbit.earth_radius_km(lat_deg)
    return cast(np.ndarray, r0 / np.linalg.norm(r0) * earth_r)


def offset_ecef(
    origin: np.ndarray,
    lat_deg: float,
    along_km: float,
    cross_km: float,
    heading_deg: float,
) -> np.ndarray:
    """Offset a surface point in the along-track / cross-track tangent plane.

    The result is reprojected onto the sphere of radius ``|origin|``.

    Args:
        origin: Surface ECEF point in kilometres. # np.ndarray[float64, (3,)]
        lat_deg: Geocentric latitude in degrees (for the local north/east basis).
        along_km: Along-track offset in kilometres (positive forward).
        cross_km: Cross-track offset in kilometres (positive to the right).
        heading_deg: Ground-track azimuth from north in degrees.

    Returns:
        Offset surface ECEF point in kilometres. # np.ndarray[float64, (3,)]
    """
    radius = float(np.linalg.norm(origin))
    lat = math.radians(lat_deg)
    lon = math.atan2(float(origin[1]), float(origin[0]))
    north = np.array(
        [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)]
    )
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    az = math.radians(heading_deg)
    along_hat = math.cos(az) * north + math.sin(az) * east
    cross_hat = -math.sin(az) * north + math.cos(az) * east
    vec = origin + along_hat * along_km + cross_hat * cross_km
    return cast(np.ndarray, vec / np.linalg.norm(vec) * radius)
