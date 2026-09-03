"""Tests for boresight-relative pointing geometry."""

from flight.payload.gimbal import boresight_error_deg, target_displacement_px


def test_centered_target_has_zero_error() -> None:
    """A centroid at the plane center yields zero pointing error."""
    az, el = boresight_error_deg(
        centroid_px=(612.0, 512.0),
        plane_width_px=1224,
        plane_height_px=1024,
        ifov_band_deg_per_px=0.002636,
    )
    assert az == 0.0
    assert el == 0.0


def test_offsets_map_through_ifov_with_image_sign_convention() -> None:
    """+x offset -> +az; +y (downward) offset -> -el; scaled by band IFOV."""
    ifov = 0.002636
    az, el = boresight_error_deg(
        centroid_px=(712.0, 412.0),
        plane_width_px=1224,
        plane_height_px=1024,
        ifov_band_deg_per_px=ifov,
    )
    assert abs(az - 100.0 * ifov) < 1e-12  # (712-612) * ifov
    assert abs(el - 100.0 * ifov) < 1e-12  # -(412-512) * ifov


def test_displacement_is_band_plane_euclidean_pixels() -> None:
    """Displacement is Euclidean distance from boresight in band-plane pixels."""
    d = target_displacement_px(
        centroid_px=(712.0, 412.0),
        plane_width_px=1224,
        plane_height_px=1024,
    )
    expected = (2.0 * 100.0**2) ** 0.5
    assert abs(d - expected) < 1e-9
