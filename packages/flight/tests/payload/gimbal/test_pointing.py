"""Tests for boresight-relative pointing geometry."""

from flight.payload.gimbal import area_weighted_com_px, boresight_error_deg, target_displacement_px


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


def test_area_weighted_com_is_area_mean() -> None:
    """Two blobs combine in proportion to pixel_area; empty input returns None."""
    from flight.libs.messages import BlobMeta

    assert area_weighted_com_px(()) is None
    a = BlobMeta(
        blob_id=1,
        bbox=(0, 0, 10, 10),
        centroid_raw=(0.0, 0.0),
        pixel_area=1,
        mean_confidence=0.9,
        persistence_count=1,
    )
    b = BlobMeta(
        blob_id=2,
        bbox=(20, 20, 30, 30),
        centroid_raw=(10.0, 20.0),
        pixel_area=3,
        mean_confidence=0.9,
        persistence_count=1,
    )
    com = area_weighted_com_px((a, b))
    assert com is not None
    assert abs(com[0] - 7.5) < 1e-9
    assert abs(com[1] - 15.0) < 1e-9
