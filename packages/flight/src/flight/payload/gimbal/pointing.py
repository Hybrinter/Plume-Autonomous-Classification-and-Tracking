"""Boresight-relative pointing geometry: tensor pixels -> angular error.

Replaces the absolute-centroid * PIXEL_TO_DEG bug (baseline Section 4.4 of the parity
baseline): error is measured FROM THE PLANE CENTER (boresight), after inverting the
preprocess crop/decimation transform, and converted to degrees via the sensor IFOV.
Sign convention: image +x (column) -> +azimuth; image +y (row, downward) -> -elevation.
The returned error is the target's angular offset from boresight -- the slew needed to
center it has the same sign.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# stdlib
import math


def _full_frame_px(
    centroid_px: tuple[float, float],
    crop_origin_px: tuple[int, int],
    scale_factor: float,
) -> tuple[float, float]:
    """Invert the crop/scale transform: tensor pixel -> full-plane pixel (float).

    Inputs:
        centroid_px: (x, y) centroid in the model-input tensor coordinate space.
        crop_origin_px: (x, y) top-left corner of the crop in the full band plane,
            in full-plane pixels. Zero for un-cropped (search mode).
        scale_factor: The decimation factor applied during preprocessing
            (0.5 for search-mode 2x decimation; 1.0 for full-resolution crop).

    Outputs:
        tuple[float, float]: (x, y) centroid in full band-plane pixel coordinates.

    Notes:
        Inverts the transform: tensor_px = (full_px - crop_origin) * scale_factor.
        Division by scale_factor is safe because scale_factor is always > 0 by
        construction (preprocess module guarantees this).
    """
    return (
        crop_origin_px[0] + centroid_px[0] / scale_factor,
        crop_origin_px[1] + centroid_px[1] / scale_factor,
    )


def boresight_error_deg(
    centroid_px: tuple[float, float],
    crop_origin_px: tuple[int, int],
    scale_factor: float,
    plane_width_px: int,
    plane_height_px: int,
    ifov_deg_per_px: float,
) -> tuple[float, float]:
    """Angular (az, el) offset of a detected centroid from the boresight, in degrees.

    The boresight corresponds to the center of the full band plane
    (plane_width_px / 2, plane_height_px / 2). Positive azimuth is to the right
    (image +x); positive elevation is upward (image -y).

    Inputs:
        centroid_px: (x, y) centroid in tensor/model-input pixel coordinates.
        crop_origin_px: (x, y) top-left of the preprocessing crop in full-plane pixels.
        scale_factor: Decimation scale applied during preprocessing (e.g. 0.5 for 2x
            decimation in search mode, 1.0 for full-resolution crop in tracking mode).
        plane_width_px: Width of the full band plane in pixels (e.g. 512).
        plane_height_px: Height of the full band plane in pixels (e.g. 512).
        ifov_deg_per_px: Instantaneous field of view per pixel in degrees (e.g. 0.02).

    Outputs:
        tuple[float, float]: (az_error_deg, el_error_deg) -- the angular offset of the
            target from boresight per the module sign convention.

    Notes:
        The centroid is first backprojected to full-plane coordinates via
        _full_frame_px, then the boresight offset is multiplied by the IFOV.
        A positive az error means the target is to the right of boresight; the gimbal
        must slew +az to center it (same sign as the error).
    """
    full_x, full_y = _full_frame_px(centroid_px, crop_origin_px, scale_factor)
    az_err = (full_x - plane_width_px / 2.0) * ifov_deg_per_px
    el_err = -(full_y - plane_height_px / 2.0) * ifov_deg_per_px
    return (az_err, el_err)


def target_displacement_px(
    centroid_px: tuple[float, float],
    crop_origin_px: tuple[int, int],
    scale_factor: float,
    plane_width_px: int,
    plane_height_px: int,
) -> float:
    """Euclidean full-plane pixel distance of the centroid from boresight (deadband input).

    Inputs:
        centroid_px: (x, y) centroid in tensor/model-input pixel coordinates.
        crop_origin_px: (x, y) top-left of the preprocessing crop in full-plane pixels.
        scale_factor: Decimation scale applied during preprocessing.
        plane_width_px: Width of the full band plane in pixels.
        plane_height_px: Height of the full band plane in pixels.

    Outputs:
        float: Euclidean distance in full-plane pixels from boresight to the centroid.
            Used as input to the deadband gate (check_deadband) in the control pipeline.

    Notes:
        The displacement is in full-plane pixels regardless of whether the frame was
        cropped or decimated, because the deadband thresholds are defined in full-plane
        pixel units for consistency across modes.
    """
    full_x, full_y = _full_frame_px(centroid_px, crop_origin_px, scale_factor)
    return math.hypot(full_x - plane_width_px / 2.0, full_y - plane_height_px / 2.0)
