"""GSD-conditioned model metadata markers and reference coverage."""

from __future__ import annotations

# stdlib
import math
from dataclasses import dataclass

# third-party
import numpy as np

# internal
from flight.libs.config import PactConfig
from flight.payload.gimbal.footprint import nominal_iss_state, tile_gsd_grid
from flight.payload.gimbal.intersect import CameraGeometry
from flight.payload.inference.contract import verify_conditioned_shapes

CONDITIONING_ID = "film-log-gsd-v1"
GSD_ENCODING = "ln_metres_over_reference_lateral_along"

__all__ = [
    "CONDITIONING_ID",
    "GSD_ENCODING",
    "GsdCoverage",
    "coverage_ok",
    "required_gsd_coverage",
    "verify_conditioned_shapes",
]


@dataclass(frozen=True, slots=True)
class GsdCoverage:
    """Per-axis metres at the reference altitude, including all tile centers."""

    minimum_m: tuple[float, float]
    maximum_m: tuple[float, float]
    altitude_m: float


def required_gsd_coverage(
    cfg: PactConfig | None = None,
    altitude_m: float = 460_000.0,
) -> GsdCoverage | None:
    """Calculate the endpoint envelope from flight geometry at a reference altitude.

    This is not a claim about arbitrary ISS altitudes. Provenance retains the
    reference altitude qualification.
    """
    cfg = PactConfig() if cfg is None else cfg
    sensor, ephemeris = cfg.sensor, cfg.ephemeris
    state = nominal_iss_state(altitude_m, ephemeris.wgs84_a_m, ephemeris.mu_m3_s2)
    if state is None:
        return None
    r, v, utc = state
    camera = CameraGeometry(
        sensor.width_px,
        sensor.height_px,
        sensor.pixel_um * 1e-6,
        sensor.optics.focal_length_mm * 1e-3,
    )
    grids = []
    for elevation in (cfg.gimbal.el_science_min_deg, cfg.gimbal.el_science_max_deg):
        grid = tile_gsd_grid(
            math.radians(elevation),
            r,
            v,
            utc,
            utc,
            ephemeris.omega_earth_rad_s,
            ephemeris.wgs84_a_m,
            ephemeris.wgs84_f,
            camera,
            grid=(cfg.inference.tile_rows, cfg.inference.tile_cols),
        )
        if grid is None:
            return None
        grids.append(grid.reshape(-1, 2))
    values = np.concatenate(grids)
    lo, hi = values.min(axis=0), values.max(axis=0)
    return GsdCoverage(
        (float(lo[0]), float(lo[1])),
        (float(hi[0]), float(hi[1])),
        altitude_m,
    )


def coverage_ok(
    minimum_m: tuple[float, float],
    maximum_m: tuple[float, float],
    required: GsdCoverage,
) -> bool:
    """Require both training axes to span the envelope, with 1 mm numeric tolerance."""
    if not all(math.isfinite(value) and value > 0 for value in (*minimum_m, *maximum_m)):
        return False
    return all(
        low <= high and low <= required_low + 1e-3 and high >= required_high - 1e-3
        for low, high, required_low, required_high in zip(
            minimum_m,
            maximum_m,
            required.minimum_m,
            required.maximum_m,
            strict=True,
        )
    )
