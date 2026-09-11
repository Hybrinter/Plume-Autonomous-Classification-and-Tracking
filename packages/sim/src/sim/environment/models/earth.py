"""Earth models: WGS-84 ellipsoid and a mean-radius sphere.

The ellipsoid path calls flight geo.wgs84_intersect_at_height so closed-loop
truth matches the payload tracking surface.
"""

from __future__ import annotations

# stdlib
import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# third-party
import numpy as np

# internal
from flight.payload.gimbal.geo import wgs84_intersect, wgs84_intersect_at_height


@runtime_checkable
class EarthModel(Protocol):
    """Height-proxy Earth intersect used by look and plume placement."""

    def intersect_at_height(
        self,
        origin_ecef_m: np.ndarray,
        unit_ecef: np.ndarray,
        height_m: float,
    ) -> tuple[np.ndarray, float] | None:
        """Return (hit_ecef_m, slant_m) or None."""
        ...


@dataclass(frozen=True, slots=True)
class Wgs84Ellipsoid:
    """Flight WGS-84 ellipsoid with uniform height-proxy inflation."""

    a_m: float
    f: float

    def intersect_at_height(
        self,
        origin_ecef_m: np.ndarray,
        unit_ecef: np.ndarray,
        height_m: float,
    ) -> tuple[np.ndarray, float] | None:
        """Forward intersect with the inflated WGS-84 ellipsoid."""
        return wgs84_intersect_at_height(origin_ecef_m, unit_ecef, self.a_m, self.f, height_m)


@dataclass(frozen=True, slots=True)
class SphereEarth:
    """Mean-radius sphere. Do not mix with flight WGS-84 intersect in one study."""

    a_m: float
    f: float = 0.0

    @property
    def radius_m(self) -> float:
        """Return the sphere radius (semi-major, flattening ignored)."""
        return self.a_m

    def intersect_at_height(
        self,
        origin_ecef_m: np.ndarray,
        unit_ecef: np.ndarray,
        height_m: float,
    ) -> tuple[np.ndarray, float] | None:
        """Intersect a sphere of radius a_m + height_m."""
        radius = self.a_m + max(height_m, 0.0)
        # Reuse ellipsoid intersect with f=0 (a == b).
        return wgs84_intersect(origin_ecef_m, unit_ecef, radius, 0.0)


def geocentric_radius_m(a_m: float, f: float, lat_rad: float) -> float:
    """Return WGS-84 geocentric radius at geocentric latitude lat_rad."""
    b_m = a_m * (1.0 - f)
    c_lat, s_lat = math.cos(lat_rad), math.sin(lat_rad)
    num = (a_m**2 * c_lat) ** 2 + (b_m**2 * s_lat) ** 2
    den = (a_m * c_lat) ** 2 + (b_m * s_lat) ** 2
    return math.sqrt(num / den)
