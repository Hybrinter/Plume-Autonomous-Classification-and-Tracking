"""Boresight-relative pointing geometry: band-plane pixels -> angular error.

Error is measured from the plane center (boresight) and converted to degrees via
the band-plane IFOV. Sign convention: image +x (column) -> +azimuth; image +y
(row, downward) -> -elevation. The returned error is the target's angular offset
from boresight -- the slew needed to center it has the same sign.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# stdlib
import math


def boresight_error_deg(
    centroid_px: tuple[float, float],
    plane_width_px: int,
    plane_height_px: int,
    ifov_band_deg_per_px: float,
) -> tuple[float, float]:
    """Angular (az, el) offset of a detected centroid from the boresight, in degrees.

    The boresight corresponds to the center of the band plane
    (plane_width_px / 2, plane_height_px / 2). Positive azimuth is to the right
    (image +x); positive elevation is upward (image -y).

    Inputs:
        centroid_px: (x, y) centroid in band-plane pixel coordinates.
        plane_width_px: Width of the band plane in pixels.
        plane_height_px: Height of the band plane in pixels.
        ifov_band_deg_per_px: Instantaneous field of view per band-plane pixel, degrees.

    Outputs:
        tuple[float, float]: (az_error_deg, el_error_deg) -- the angular offset of the
            target from boresight per the module sign convention.
    """
    az_err = (centroid_px[0] - plane_width_px / 2.0) * ifov_band_deg_per_px
    el_err = -(centroid_px[1] - plane_height_px / 2.0) * ifov_band_deg_per_px
    return (az_err, el_err)


def target_displacement_px(
    centroid_px: tuple[float, float],
    plane_width_px: int,
    plane_height_px: int,
) -> float:
    """Euclidean band-plane pixel distance of the centroid from boresight.

    Inputs:
        centroid_px: (x, y) centroid in band-plane pixel coordinates.
        plane_width_px: Width of the band plane in pixels.
        plane_height_px: Height of the band plane in pixels.

    Outputs:
        float: Euclidean distance in band-plane pixels from boresight to the centroid.
    """
    return math.hypot(centroid_px[0] - plane_width_px / 2.0, centroid_px[1] - plane_height_px / 2.0)
