"""Tests for boresight-relative pointing geometry."""

from flight.payload.gimbal import boresight_error_deg, target_displacement_px


def test_centered_target_has_zero_error() -> None:
    """A centroid at the plane center yields zero pointing error (the silent-wrongness fix)."""
    az, el = boresight_error_deg(
        centroid_px=(256.0, 256.0),
        crop_origin_px=(0, 0),
        scale_factor=1.0,
        plane_width_px=512,
        plane_height_px=512,
        ifov_deg_per_px=0.02,
    )
    assert az == 0.0
    assert el == 0.0


def test_offsets_map_through_ifov_with_image_sign_convention() -> None:
    """+x offset -> +az; +y (downward) offset -> -el; scaled by IFOV."""
    az, el = boresight_error_deg(
        centroid_px=(306.0, 206.0),
        crop_origin_px=(0, 0),
        scale_factor=1.0,
        plane_width_px=512,
        plane_height_px=512,
        ifov_deg_per_px=0.02,
    )
    assert abs(az - 1.0) < 1e-9  # (306-256) * 0.02
    assert abs(el - 1.0) < 1e-9  # -(206-256) * 0.02


def test_crop_and_scale_backproject_before_conversion() -> None:
    """Crop origin and decimation scale are inverted before the angular conversion."""
    az, el = boresight_error_deg(
        centroid_px=(85.0, 85.0),
        crop_origin_px=(0, 0),
        scale_factor=0.5,  # decimated search mode: tensor px = plane px * 0.5
        plane_width_px=512,
        plane_height_px=512,
        ifov_deg_per_px=0.02,
    )
    assert abs(az - (170.0 - 256.0) * 0.02) < 1e-9
    assert abs(el - (-(170.0 - 256.0) * 0.02)) < 1e-9


def test_displacement_is_full_frame_euclidean_pixels() -> None:
    """Deadband displacement is measured in full-frame plane pixels."""
    d = target_displacement_px(
        centroid_px=(85.0, 85.0),
        crop_origin_px=(0, 0),
        scale_factor=0.5,
        plane_width_px=512,
        plane_height_px=512,
    )
    expected = (2.0 * (170.0 - 256.0) ** 2) ** 0.5
    assert abs(d - expected) < 1e-9
