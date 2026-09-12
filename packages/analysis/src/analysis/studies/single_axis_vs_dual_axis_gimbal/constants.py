"""WGS-84 and Earth-rotation constants for the gimbal study.

Values come from flight ``EphemerisConfig`` (SI) converted to km. Mean
spherical radius is study-only, for haversine clustering.

Contains:
  - MU_KM3_S2 / OMEGA_EARTH_RAD_S from EphemerisConfig.
  - WGS84_A_KM / WGS84_B_KM from EphemerisConfig.
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
