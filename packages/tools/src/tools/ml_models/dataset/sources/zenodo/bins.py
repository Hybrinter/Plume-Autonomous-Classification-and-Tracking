"""Native Zenodo and discrete reference-orbit flight GSD bins."""

from __future__ import annotations

import math

from flight.libs.config import EphemerisConfig, SensorConfig
from flight.payload.gimbal.footprint import GsdPair, nominal_iss_state, pixel_gsd, tile_gsd_grid
from flight.payload.gimbal.intersect import CameraGeometry

from tools.ml_models.dataset.raw import BinSpec

EXTENT_M = 1200.0
NATIVE_SIDE = 120
FLIGHT_ELEVATIONS_DEG = (5.0, 15.0, 25.0, 35.0, 45.0)


def make_bins(
    altitude_m: float = 460_000.0,
    sensor: SensorConfig | None = None,
    ephemeris: EphemerisConfig | None = None,
) -> tuple[BinSpec, ...]:
    """Return native 10 m plus boresight bins from flight's optical/Earth geometry."""
    sensor = SensorConfig() if sensor is None else sensor
    ephemeris = EphemerisConfig() if ephemeris is None else ephemeris
    state = nominal_iss_state(altitude_m, ephemeris.wgs84_a_m)
    if state is None:
        raise ValueError("invalid reference orbit for Zenodo GSD bins")
    r, v, utc = state
    camera = CameraGeometry(
        sensor.width_px,
        sensor.height_px,
        sensor.pixel_um * 1e-6,
        sensor.optics.focal_length_mm * 1e-3,
    )
    bins = [BinSpec("native10", 10.0, 10.0)]
    for elevation in FLIGHT_ELEVATIONS_DEG:
        pair = pixel_gsd(
            camera.width_px / 2,
            camera.height_px / 2,
            math.radians(elevation),
            r,
            v,
            utc,
            utc,
            ephemeris.omega_earth_rad_s,
            ephemeris.wgs84_a_m,
            ephemeris.wgs84_f,
            camera,
        )
        if pair is None:
            raise ValueError("reference GSD ray missed Earth")
        if elevation == FLIGHT_ELEVATIONS_DEG[-1]:
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
            )
            if grid is None:
                raise ValueError("reference tile GSD grid missed Earth")
            # The endpoint must cover frame edges, even after output-size rounding.
            lateral_max = float(grid[..., 0].max())
            along_max = float(grid[..., 1].max())
            width = math.floor(EXTENT_M / lateral_max)
            height = math.floor(EXTENT_M / along_max)
            if min(height, width) < 1:
                raise ValueError("reference tile footprint exceeds source extent")
            pair = GsdPair(EXTENT_M / width, EXTENT_M / height)
        bins.append(
            BinSpec(
                f"elevation{int(elevation)}",
                pair.lateral_m,
                pair.along_m,
                elevation,
            )
        )
    return tuple(bins)


def bin_hw(bin_spec: BinSpec) -> tuple[int, int]:
    """Round a fixed 1200 m ground window to H along-track, W lateral."""
    if not all(
        math.isfinite(value) and value >= 10 for value in (bin_spec.lateral_m, bin_spec.along_m)
    ):
        raise ValueError("Zenodo bins may only retain or coarsen native 10 m data")
    return round(EXTENT_M / bin_spec.along_m), round(EXTENT_M / bin_spec.lateral_m)


def actual_gsd(bin_spec: BinSpec) -> GsdPair:
    """Record metres per output pixel after rounding, not the requested target."""
    height, width = bin_hw(bin_spec)
    if min(height, width) < 1:
        raise ValueError("empty output bin")
    return GsdPair(EXTENT_M / width, EXTENT_M / height)


DEFAULT_BINS = make_bins()
