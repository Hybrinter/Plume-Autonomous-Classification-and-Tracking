"""Tests for per-pixel and per-tile GSD on the Earth ellipsoid."""

import math

import numpy as np
import pytest
from flight.libs.config import EphemerisConfig, SensorConfig
from flight.libs.types import Err
from flight.payload.gimbal.footprint import (
    GSD_REFERENCE_M,
    GsdPair,
    nominal_iss_state,
    pixel_gsd,
    tile_gsd_grid,
    to_model_gsd,
)
from flight.payload.gimbal.intersect import CameraGeometry

_ALTITUDE_M = 460_000.0


def _camera() -> CameraGeometry:
    """Return the configured sensor geometry."""
    sensor = SensorConfig()
    return CameraGeometry(
        sensor.width_px,
        sensor.height_px,
        sensor.pixel_um * 1e-6,
        sensor.optics.focal_length_mm * 1e-3,
    )


def _state() -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Return the nominal ISS reference orbit, which must always exist."""
    state = nominal_iss_state(_ALTITUDE_M)
    assert state is not None
    return state


def _pixel(theta_g_deg: float, u: float, v: float) -> GsdPair | None:
    """Evaluate pixel_gsd at one pixel for the reference orbit."""
    r, v_eci, utc = _state()
    ephemeris = EphemerisConfig()
    return pixel_gsd(
        u,
        v,
        math.radians(theta_g_deg),
        r,
        v_eci,
        utc,
        utc,
        ephemeris.omega_earth_rad_s,
        ephemeris.wgs84_a_m,
        ephemeris.wgs84_f,
        _camera(),
    )


def test_nadir_center_matches_paraxial_reference() -> None:
    """Boresight nadir GSD is the 15.87 m paraxial reference."""
    pair = _pixel(0.0, 1032.0, 772.0)
    assert pair is not None
    assert pair.lateral_m == pytest.approx(15.87, rel=0.02)
    assert pair.along_m == pytest.approx(15.87, rel=0.02)


def test_boresight_at_45_degrees() -> None:
    """The 45 degree boresight footprint is anisotropic."""
    pair = _pixel(45.0, 1032.0, 772.0)
    assert pair is not None
    assert pair.lateral_m == pytest.approx(23.318458, rel=0.02)
    assert pair.along_m == pytest.approx(35.757193, rel=0.02)


def test_grid_decreases_down_h_at_45() -> None:
    """Image down looks back: along-track GSD shrinks toward the bottom row."""
    r, v_eci, utc = _state()
    ephemeris = EphemerisConfig()
    grid = tile_gsd_grid(
        math.radians(45.0),
        r,
        v_eci,
        utc,
        utc,
        ephemeris.omega_earth_rad_s,
        ephemeris.wgs84_a_m,
        ephemeris.wgs84_f,
        _camera(),
    )
    assert grid is not None
    along_col4 = grid[:, 4, 1]
    assert np.all(np.diff(along_col4.astype(np.float64)) < 0.0)
    assert float(along_col4[0]) == pytest.approx(37.96, rel=0.02)
    assert float(along_col4[-1]) == pytest.approx(33.76, rel=0.02)


def test_ray_miss_returns_none() -> None:
    """A boresight beyond the horizon returns None, not a fault."""
    assert _pixel(80.0, 1032.0, 772.0) is None


def test_invalid_geometry_returns_none() -> None:
    """Non-finite scalars and bad cameras return None."""
    assert _pixel(0.0, float("nan"), 772.0) is None
    r, v_eci, utc = _state()
    ephemeris = EphemerisConfig()
    bad_camera = CameraGeometry(2064, 1544, 0.0, 0.1)
    assert (
        pixel_gsd(
            1032.0,
            772.0,
            0.0,
            r,
            v_eci,
            utc,
            utc,
            ephemeris.omega_earth_rad_s,
            ephemeris.wgs84_a_m,
            ephemeris.wgs84_f,
            bad_camera,
        )
        is None
    )
    assert nominal_iss_state(0.0) is None
    assert nominal_iss_state(float("inf")) is None


def test_grid_rejects_indivisible_shape() -> None:
    """A grid that does not divide the frame returns None."""
    r, v_eci, utc = _state()
    ephemeris = EphemerisConfig()
    assert (
        tile_gsd_grid(
            0.0,
            r,
            v_eci,
            utc,
            utc,
            ephemeris.omega_earth_rad_s,
            ephemeris.wgs84_a_m,
            ephemeris.wgs84_f,
            _camera(),
            grid=(7, 8),
        )
        is None
    )


@pytest.mark.parametrize("grid", [(0, 8), (8, 0), (True, 8), (8.0, 8), (8,), [8, 8]])
def test_grid_rejects_malformed_grid(grid: object) -> None:
    """Invalid grid types and dimensions return None rather than raising."""
    r, v_eci, utc = _state()
    ephemeris = EphemerisConfig()
    assert (
        tile_gsd_grid(
            0.0,
            r,
            v_eci,
            utc,
            utc,
            ephemeris.omega_earth_rad_s,
            ephemeris.wgs84_a_m,
            ephemeris.wgs84_f,
            _camera(),
            grid=grid,  # type: ignore[arg-type]
        )
        is None
    )


def test_model_gsd_encoding_result() -> None:
    """The reference encodes to zero; malformed input is an Err."""
    encoded = to_model_gsd(np.array([15.87, 15.87]), GSD_REFERENCE_M)
    assert not isinstance(encoded, Err)
    np.testing.assert_allclose(encoded.value, np.zeros(2, dtype=np.float32), atol=1e-6)
    assert isinstance(to_model_gsd(np.array([-1.0, 15.87]), 15.87), Err)
    assert isinstance(to_model_gsd(np.zeros((2, 3)), 15.87), Err)
    assert isinstance(to_model_gsd(np.array([15.87, 15.87]), 0.0), Err)


def test_model_gsd_encoding_handles_extreme_finite_values() -> None:
    """Finite float64 extremes stay finite after log-ratio encoding to float32."""
    values = np.array([np.nextafter(0.0, 1.0), np.finfo(np.float64).max])
    encoded = to_model_gsd(values, 15.87)
    assert not isinstance(encoded, Err)
    assert np.all(np.isfinite(encoded.value))


@pytest.mark.parametrize(
    "values",
    [np.array(["unknown", "unknown"]), np.array([1j, 2j]), np.array([True, True])],
)
def test_model_gsd_rejects_nonreal_measurements(values: np.ndarray) -> None:
    """Nonphysical array domains return a fault without casting or raising."""
    assert isinstance(to_model_gsd(values), Err)
