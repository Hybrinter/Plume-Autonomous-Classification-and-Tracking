"""WGS-84 and Earth-rotation constants shared by design studies.

Values are the km/deg view of flight ``EphemerisConfig`` SI constants.
Study-specific TLE elements, gimbal boxes, and optics live in each study's
assumptions module.

Contains:
  - MU_KM3_S2: Earth gravitational parameter.
  - OMEGA_EARTH_RAD_S: Earth inertial rotation rate.
  - WGS84_A_KM / WGS84_B_KM: WGS-84 ellipsoid semi-axes.
  - MEAN_EARTH_RADIUS_KM: spherical Earth used for haversine clustering.
"""

from __future__ import annotations

from flight.libs.config import EphemerisConfig

_EPH = EphemerisConfig()

MU_KM3_S2: float = _EPH.mu_m3_s2 / 1.0e9
OMEGA_EARTH_RAD_S: float = _EPH.omega_earth_rad_s
WGS84_A_KM: float = _EPH.wgs84_a_m / 1000.0
WGS84_B_KM: float = _EPH.wgs84_a_m * (1.0 - _EPH.wgs84_f) / 1000.0
MEAN_EARTH_RADIUS_KM: float = 6371.0
