"""Tests for pinhole boresight-relative pointing geometry."""

import math

from flight.payload.gimbal import boresight_error_deg, pinhole_error_rad, target_displacement_px

_PITCH_M = 2.0 * 3.45e-6
_FOCAL_M = 0.150


def test_centered_target_has_zero_error() -> None:
    """A centroid at the plane center yields zero pointing error."""
    az, el = boresight_error_deg(
        centroid_px=(612.0, 512.0),
        plane_width_px=1224,
        plane_height_px=1024,
        pixel_pitch_m=_PITCH_M,
        focal_m=_FOCAL_M,
    )
    assert abs(az) < 1e-12
    assert abs(el) < 1e-12


def test_offsets_map_through_pinhole_with_image_sign_convention() -> None:
    """+x offset -> +az; +y (downward) offset -> -el via atan(offset * pitch / f)."""
    az, el = pinhole_error_rad(
        centroid_px=(712.0, 412.0),
        plane_width_px=1224,
        plane_height_px=1024,
        pixel_pitch_m=_PITCH_M,
        focal_m=_FOCAL_M,
    )
    expected_az = math.atan2(100.0 * _PITCH_M, _FOCAL_M)
    expected_el = math.atan2(100.0 * _PITCH_M, _FOCAL_M)
    assert abs(az - expected_az) < 1e-12
    assert abs(el - expected_el) < 1e-12


def test_displacement_is_band_plane_euclidean_pixels() -> None:
    """Displacement is Euclidean distance from boresight in band-plane pixels."""
    d = target_displacement_px(
        centroid_px=(712.0, 412.0),
        plane_width_px=1224,
        plane_height_px=1024,
    )
    expected = (2.0 * 100.0**2) ** 0.5
    assert abs(d - expected) < 1e-9
