"""
ROI crop, resampling, and coordinate back-projection for PACT inference.

Satisfies: REQ-AIML-PREP-003

Maps the full band plane (1024 x 1224 px at 19.3 m/px for the flight IMX264 + 150 mm
geometry) onto the fixed model input (256 x 256) and records the transform so that a
detection in tensor pixels can be back-projected to full-plane pixels for the gimbal
delta command. Two modes:

    Search (decimate_to_size): the model needs to see the whole plane for a presence
    sweep. The plane is centre-cropped to an integer multiple of the output size, then
    AREA-AVERAGED (box mean) down to the output. Area averaging is a low-pass filter
    matched to the decimation factor; the IMX264 mosaic is roughly Nyquist-sampled by
    the 150 mm f/4 lens (diffraction spot 4.8-8.2 um vs 3.45 um pitch), so bare stride
    slicing aliases plume edges into spurious high-frequency structure.

    Track (crop_and_upsample): a small window around the Kalman-estimated target is
    cropped at full plane resolution and cubic-upsampled 2x to the output size, which
    brings the effective sampling to ~9.7 m/px, matching the ~10 m Sentinel-2 sampling
    the model was trained on. Cubic interpolation can ring at sharp plume edges; the
    output is clipped to the input value range to bound the overshoot.

Transform convention (shared with flight.payload.gimbal.pointing):
    tensor_px = (plane_px - crop_origin_px) * scale_factor
This linear map ignores the half-pixel centre shift of block averaging and zooming
((factor - 1) / 2 plane px, i.e. 1.5 px = 0.004 deg at 4x); that bias is far inside the
0.1 deg pointing budget and keeping the map linear keeps the inverse exact.

All functions are pure: they perform no I/O and have no side effects.

Contains:
  - RoiTransform: crop origin and scale of a tensor relative to the full plane.
  - crop_plane: clamped, size-checked ROI crop; Err(FRAME_MALFORMED) on bad sizes.
  - crop_to_roi: legacy crop returning a tuple; oversized requests shrink to the plane.
  - decimate_area: anti-aliased integer-factor decimation by box mean.
  - decimate_to_size: centre-crop plus decimate_area to an exact output size.
  - upsample: integer-factor spline upsampling (order 3 = cubic), range-clipped.
  - crop_and_upsample: full-resolution crop plus upsample to an exact output size.
  - plane_to_tensor_px / tensor_to_plane_px: exact float transform and inverse.
  - backproject_pixel: legacy integer inverse.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass

# third-party
import numpy as np
import scipy.ndimage

# internal
from flight.libs.types import Err, FaultCode, Ok, Result


@dataclass(frozen=True, slots=True)
class RoiTransform:
    """Geometry of an inference tensor relative to the full band plane.

    Attributes:
        crop_origin_px: tuple[int, int] (x, y) full-plane pixel of the tensor's
            top-left corner.
        scale_factor: float tensor pixels per plane pixel (0.25 for 4x decimation,
            2.0 for 2x upsampling, 1.0 for a plain crop). Always > 0.
    """

    crop_origin_px: tuple[int, int]
    scale_factor: float


def _crop_bounds(
    center: float,
    size_out: int,
    size_in: int,
) -> int:
    """Clamped start index of a size_out window centred at center within size_in.

    Inputs:
        center (float): Desired window centre, pixels.
        size_out (int): Window length, pixels (<= size_in).
        size_in (int): Axis length, pixels.

    Outputs:
        int: Start index in [0, size_in - size_out].
    """
    start = round(center) - size_out // 2
    return max(0, min(start, size_in - size_out))


def crop_plane(
    bands: np.ndarray,
    center_px: tuple[float, float],
    output_size: tuple[int, int],
) -> Result[tuple[np.ndarray, tuple[int, int]], FaultCode]:
    """Crop a (C, H, W) array to a fixed-size ROI centred at center_px, clamped to the plane.

    Inputs:
        bands (np.ndarray[float32, (C, H, W)]): Full-plane multispectral array.
        center_px (tuple[float, float]): (x, y) crop centre in full-plane pixels; x is
            the column (width) axis, y the row (height) axis. Rounded to the nearest
            pixel (not truncated) so the window is not biased toward the origin.
        output_size (tuple[int, int]): (H_out, W_out); each must be in [1, H] / [1, W].

    Outputs:
        Result[tuple[np.ndarray, tuple[int, int]], FaultCode]:
            Ok((cropped, crop_origin)) with cropped np.ndarray[float32, (C, H_out,
                W_out)] and crop_origin the (x, y) full-plane pixel of its top-left
                corner (differs from the centred position when clamping applied);
            Err(FaultCode.FRAME_MALFORMED) if bands is not rank 3 or output_size is
                non-positive or exceeds the plane.
    """
    if bands.ndim != 3:
        return Err(FaultCode.FRAME_MALFORMED)
    _c, h, w = bands.shape
    h_out, w_out = output_size
    if h_out < 1 or w_out < 1 or h_out > h or w_out > w:
        return Err(FaultCode.FRAME_MALFORMED)
    cx, cy = center_px
    x0 = _crop_bounds(cx, w_out, w)
    y0 = _crop_bounds(cy, h_out, h)
    cropped = bands[:, y0 : y0 + h_out, x0 : x0 + w_out]  # np.ndarray[float32, (C, Ho, Wo)]
    return Ok((cropped, (x0, y0)))


def crop_to_roi(
    bands: np.ndarray,  # (C, H, W) float32
    center_px: tuple[int, int],
    output_size: tuple[int, int],  # (H_out, W_out)
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop a multispectral array to a fixed-size ROI centred at center_px.

    Legacy entry point with an unchecked return type; delegates to crop_plane. A
    requested size larger than the plane is shrunk to the plane on that axis (instead of
    producing an empty or mis-shaped slice), so the result is always a valid array.

    Inputs:
        bands (np.ndarray[float32, (C, H, W)]): Full-frame multispectral array.
        center_px (tuple[int, int]): (x, y) pixel coordinate of the crop centre in
            full-frame space.
        output_size (tuple[int, int]): Desired (H_out, W_out) of the cropped output.

    Outputs:
        tuple[np.ndarray, tuple[int, int]]: (cropped_bands, crop_origin) where
            cropped_bands is np.ndarray[float32, (C, min(H_out, H), min(W_out, W))] and
            crop_origin is the (x, y) top-left corner of the crop in the full frame,
            used by backproject_pixel() / tensor_to_plane_px() to invert the crop.

    Satisfies: REQ-AIML-PREP-003
    """
    _c, h, w = bands.shape
    h_out = max(1, min(output_size[0], h))
    w_out = max(1, min(output_size[1], w))
    result = crop_plane(bands, (float(center_px[0]), float(center_px[1])), (h_out, w_out))
    if isinstance(result, Err):
        # Only reachable for a non-rank-3 input; the sizes were clamped above.
        return bands, (0, 0)
    return result.value


def decimate_area(bands: np.ndarray, factor: int) -> Result[np.ndarray, FaultCode]:
    """Decimate by an integer factor using the box mean of each factor x factor block.

    The box mean is the area-weighted low-pass filter matched to the decimation, so
    spatial frequencies above the new Nyquist limit are attenuated rather than folded
    back into the image as they are with stride slicing.

    Inputs:
        bands (np.ndarray[float32, (C, H, W)]): Input planes; H and W must be multiples
            of factor.
        factor (int): Decimation factor >= 1 (1 returns a float32 copy).

    Outputs:
        Result[np.ndarray, FaultCode]:
            Ok(np.ndarray[float32, (C, H / factor, W / factor)]);
            Err(FaultCode.FRAME_MALFORMED) if bands is not rank 3, factor < 1, or H or
                W is not a multiple of factor.
    """
    if bands.ndim != 3 or factor < 1:
        return Err(FaultCode.FRAME_MALFORMED)
    c, h, w = bands.shape
    if h % factor or w % factor:
        return Err(FaultCode.FRAME_MALFORMED)
    if factor == 1:
        return Ok(bands.astype(np.float32))
    blocks = bands.reshape(c, h // factor, factor, w // factor, factor)
    return Ok(blocks.mean(axis=(2, 4), dtype=np.float32))  # np.ndarray[float32, (C, H/f, W/f)]


def decimate_to_size(
    bands: np.ndarray,
    output_size: tuple[int, int],
) -> Result[tuple[np.ndarray, RoiTransform], FaultCode]:
    """Search-mode resample: centre-crop to a multiple of output_size, then box-decimate.

    The decimation factor is the largest integer that fits on BOTH axes,
    min(H // H_out, W // W_out), so the whole output is covered by real pixels at one
    uniform scale; the plane is then centre-cropped to (factor * H_out, factor * W_out)
    and area-averaged. For the flight 1024 x 1224 plane and a 256 x 256 input this is
    factor 4 with a 1024 x 1024 centre crop that drops 100 px (0.26 deg) each side
    cross-track.

    Inputs:
        bands (np.ndarray[float32, (C, H, W)]): Full band plane.
        output_size (tuple[int, int]): (H_out, W_out) model input size.

    Outputs:
        Result[tuple[np.ndarray, RoiTransform], FaultCode]:
            Ok((tensor, transform)) with tensor np.ndarray[float32, (C, H_out, W_out)]
                and transform.scale_factor = 1 / factor;
            Err(FaultCode.FRAME_MALFORMED) if bands is not rank 3 or the plane is
                smaller than output_size on either axis.
    """
    if bands.ndim != 3:
        return Err(FaultCode.FRAME_MALFORMED)
    _c, h, w = bands.shape
    h_out, w_out = output_size
    if h_out < 1 or w_out < 1 or h < h_out or w < w_out:
        return Err(FaultCode.FRAME_MALFORMED)
    factor = min(h // h_out, w // w_out)
    crop = crop_plane(bands, (w / 2.0, h / 2.0), (factor * h_out, factor * w_out))
    if isinstance(crop, Err):
        return Err(crop.error)
    cropped, origin = crop.value
    decimated = decimate_area(cropped, factor)
    if isinstance(decimated, Err):
        return Err(decimated.error)
    return Ok((decimated.value, RoiTransform(crop_origin_px=origin, scale_factor=1.0 / factor)))


def upsample(bands: np.ndarray, factor: int, order: int = 3) -> np.ndarray:
    """Upsample each plane by an integer factor with spline interpolation.

    Uses scipy.ndimage.zoom with grid_mode=True so the output covers exactly the input
    extent (factor x more samples per axis), per band, with reflect boundary handling.
    order 3 is cubic (the PDR baseline for matching the ~10 m training sampling); order
    1 is bilinear for an artefact-free alternative. The output is clipped to the input
    [min, max] so spline overshoot cannot leave the input value range.

    Inputs:
        bands (np.ndarray[float32, (C, H, W)]): Input planes.
        factor (int): Upsampling factor >= 1 (1 returns a float32 copy).
        order (int): Spline order in [0, 5]; 3 = cubic.

    Outputs:
        np.ndarray[float32, (C, H * factor, W * factor)]: Upsampled planes.
    """
    if factor == 1:
        return bands.astype(np.float32)
    zoomed = scipy.ndimage.zoom(
        bands.astype(np.float32),
        zoom=(1, factor, factor),
        order=order,
        mode="reflect",
        grid_mode=True,
    )  # np.ndarray[float32, (C, H*f, W*f)]
    lo = float(bands.min())
    hi = float(bands.max())
    return np.clip(zoomed, lo, hi).astype(np.float32)


def crop_and_upsample(
    bands: np.ndarray,
    center_px: tuple[float, float],
    output_size: tuple[int, int],
    upsample_factor: int,
    order: int = 3,
) -> Result[tuple[np.ndarray, RoiTransform], FaultCode]:
    """Track-mode resample: crop output_size / upsample_factor at full resolution, upsample.

    For a 256 x 256 input and factor 2 this crops a 128 x 128 band-plane window (2.5 km
    at 19.3 m/px) around the target and cubic-upsamples it to 256 x 256 at ~9.7 m/px.

    Inputs:
        bands (np.ndarray[float32, (C, H, W)]): Full band plane.
        center_px (tuple[float, float]): (x, y) window centre in full-plane pixels
            (e.g. the Kalman estimate mapped through the IFOV).
        output_size (tuple[int, int]): (H_out, W_out) model input size; each must be a
            multiple of upsample_factor.
        upsample_factor (int): Integer upsampling factor >= 1.
        order (int): Spline order for upsample; 3 = cubic.

    Outputs:
        Result[tuple[np.ndarray, RoiTransform], FaultCode]:
            Ok((tensor, transform)) with tensor np.ndarray[float32, (C, H_out, W_out)]
                and transform.scale_factor = upsample_factor;
            Err(FaultCode.FRAME_MALFORMED) if upsample_factor < 1, output_size is not
                a multiple of it, or the crop window does not fit the plane.
    """
    if upsample_factor < 1:
        return Err(FaultCode.FRAME_MALFORMED)
    h_out, w_out = output_size
    if h_out % upsample_factor or w_out % upsample_factor:
        return Err(FaultCode.FRAME_MALFORMED)
    crop = crop_plane(bands, center_px, (h_out // upsample_factor, w_out // upsample_factor))
    if isinstance(crop, Err):
        return Err(crop.error)
    cropped, origin = crop.value
    tensor = upsample(cropped, upsample_factor, order)
    return Ok((tensor, RoiTransform(crop_origin_px=origin, scale_factor=float(upsample_factor))))


def plane_to_tensor_px(
    plane_px: tuple[float, float],
    transform: RoiTransform,
) -> tuple[float, float]:
    """Map a full-plane pixel to tensor pixel coordinates.

        tensor = (plane - crop_origin) * scale_factor

    Inputs:
        plane_px (tuple[float, float]): (x, y) in full band-plane pixels.
        transform (RoiTransform): Crop origin and scale of the tensor.

    Outputs:
        tuple[float, float]: (x, y) in tensor pixels (float, unrounded).
    """
    return (
        (plane_px[0] - transform.crop_origin_px[0]) * transform.scale_factor,
        (plane_px[1] - transform.crop_origin_px[1]) * transform.scale_factor,
    )


def tensor_to_plane_px(
    tensor_px: tuple[float, float],
    transform: RoiTransform,
) -> tuple[float, float]:
    """Map a tensor pixel back to full-plane pixel coordinates (exact inverse).

        plane = crop_origin + tensor / scale_factor

    Inputs:
        tensor_px (tuple[float, float]): (x, y) in tensor pixels (e.g. a blob centroid).
        transform (RoiTransform): Crop origin and scale of the tensor.

    Outputs:
        tuple[float, float]: (x, y) in full band-plane pixels (float, unrounded, so no
            quantisation is added before the IFOV conversion).
    """
    return (
        transform.crop_origin_px[0] + tensor_px[0] / transform.scale_factor,
        transform.crop_origin_px[1] + tensor_px[1] / transform.scale_factor,
    )


def backproject_pixel(
    px: tuple[int, int],
    crop_origin: tuple[int, int],
    scale_factor: float,
) -> tuple[int, int]:
    """Convert a pixel coordinate in the cropped tensor to full-frame pixel space.

    Legacy integer form of tensor_to_plane_px:
        full_x = crop_origin_x + round(px_x / scale_factor)
        full_y = crop_origin_y + round(px_y / scale_factor)
    Prefer tensor_to_plane_px for control use; rounding here quantises the centroid to
    a whole plane pixel before the angular conversion.

    Inputs:
        px (tuple[int, int]): (x, y) pixel coordinate in the cropped/scaled tensor.
        crop_origin (tuple[int, int]): (x, y) top-left corner of the crop in full-frame
            space, as returned by crop_to_roi().
        scale_factor (float): Tensor pixels per plane pixel (e.g. 0.5 means the crop
            was downsampled 2x before inference). Use 1.0 if no scaling.

    Outputs:
        tuple[int, int]: (x, y) pixel coordinate in the original full-frame space.

    Satisfies: REQ-AIML-PREP-003
    """
    full_x: int = crop_origin[0] + round(px[0] / scale_factor)
    full_y: int = crop_origin[1] + round(px[1] / scale_factor)
    return (full_x, full_y)
