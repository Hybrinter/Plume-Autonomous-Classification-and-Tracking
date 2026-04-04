"""
pact.preprocessing.crop — ROI crop and coordinate back-projection for PACT inference.

Satisfies: REQ-AIML-PREP-003

Crops the calibrated multispectral array to a fixed-size region of interest (ROI)
centred on the gimbal's current pointing position. The crop origin is preserved so
that pixel coordinates in the cropped tensor can be back-projected to full-frame space,
which is required for issuing gimbal delta commands from detected blob centroids.

All functions are pure: they perform no I/O and have no side effects.
"""

from __future__ import annotations

# third-party
import numpy as np


def crop_to_roi(
    bands: np.ndarray,              # (C, H, W) float32
    center_px: tuple[int, int],
    output_size: tuple[int, int],   # (H_out, W_out)
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop a multispectral array to a fixed-size ROI centred at center_px.

    The crop is clamped to the image boundaries if the centred window would extend
    outside the frame. The returned crop_origin reflects the actual top-left corner
    of the crop in full-frame pixel space, which may differ from the intended centre
    if clamping was applied.

    Args:
        bands:       Full-frame multispectral array, shape (C, H, W), float32.
        center_px:   (x, y) pixel coordinate of the crop centre in full-frame space.
                     x is the column index (width axis), y is the row index (height axis).
        output_size: Desired (H_out, W_out) of the cropped output. Must be <= (H, W).

    Returns:
        A tuple (cropped_bands, crop_origin) where:
        - cropped_bands: np.ndarray[float32, (C, H_out, W_out)] — the cropped array.
        - crop_origin:   (x, y) offset of the top-left corner of the crop in the full
                         frame. Used by backproject_pixel() to invert the crop transform.

    Satisfies: REQ-AIML-PREP-003
    """
    _c, H, W = bands.shape
    H_out, W_out = output_size
    cx, cy = center_px

    # Compute top-left corner of the crop window (before clamping).
    x0: int = cx - W_out // 2
    y0: int = cy - H_out // 2

    # Clamp to image boundaries.
    x0 = max(0, min(x0, W - W_out))
    y0 = max(0, min(y0, H - H_out))

    x1: int = x0 + W_out
    y1: int = y0 + H_out

    cropped: np.ndarray = bands[:, y0:y1, x0:x1]  # np.ndarray[float32, (C, H_out, W_out)]
    crop_origin: tuple[int, int] = (x0, y0)

    return (cropped, crop_origin)


def backproject_pixel(
    px: tuple[int, int],
    crop_origin: tuple[int, int],
    scale_factor: float,
) -> tuple[int, int]:
    """Convert a pixel coordinate in the cropped tensor to full-frame pixel space.

    Applies the inverse of the crop-and-scale transform applied during preprocessing:
        full_x = crop_origin_x + round(px_x / scale_factor)
        full_y = crop_origin_y + round(px_y / scale_factor)

    When scale_factor == 1.0 (no scaling, only cropping) this reduces to a simple
    translation by crop_origin.

    Args:
        px:           (x, y) pixel coordinate in the cropped/scaled tensor.
        crop_origin:  (x, y) top-left corner of the crop in full-frame space,
                      as returned by crop_to_roi().
        scale_factor: Spatial scale factor applied after cropping (e.g. 0.5 means the
                      crop was downsampled 2x before inference). Use 1.0 if no scaling.

    Returns:
        (x, y) pixel coordinate in the original full-frame space.

    Satisfies: REQ-AIML-PREP-003
    """
    full_x: int = crop_origin[0] + round(px[0] / scale_factor)
    full_y: int = crop_origin[1] + round(px[1] / scale_factor)
    return (full_x, full_y)
