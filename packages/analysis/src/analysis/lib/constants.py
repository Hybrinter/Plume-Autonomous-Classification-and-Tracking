"""WGS-84 and Earth-rotation constants shared by design studies.

These are physical constants, not study assumptions. Study-specific TLE
elements, gimbal boxes, and optics live in each study's assumptions module.

Contains:
  - MU_KM3_S2: Earth gravitational parameter.
  - OMEGA_EARTH_RAD_S: Earth inertial rotation rate.
  - WGS84_A_KM / WGS84_B_KM: WGS-84 ellipsoid semi-axes.
  - MEAN_EARTH_RADIUS_KM: spherical Earth used for haversine clustering.
"""

from __future__ import annotations

MU_KM3_S2: float = 398600.4418
OMEGA_EARTH_RAD_S: float = 7.2921159e-5
WGS84_A_KM: float = 6378.137
WGS84_B_KM: float = 6356.752314245
MEAN_EARTH_RADIUS_KM: float = 6371.0
