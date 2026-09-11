# flight.payload.preprocess.crop

**Source:** `packages/flight/src/flight/payload/preprocess/crop.py`
**Kind:** pure module

## Purpose

This module resamples the full band plane onto the fixed model input and records the
crop geometry as a `RoiTransform`, mapping detection coordinates in tensor pixels back
to full-plane pixels for the gimbal delta command. It provides a search path
(centre-crop plus area-average decimation) and a track path (full-resolution crop plus
spline upsampling).

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RoiTransform` | dataclass | Crop origin and scale of a tensor relative to the full plane |
| `crop_plane` | function | Clamped, size-checked ROI crop; `Err(FRAME_MALFORMED)` on bad sizes |
| `decimate_area` | function | Anti-aliased integer-factor decimation by box mean |
| `decimate_to_size` | function | Centre-crop plus `decimate_area` to an exact output size |
| `upsample` | function | Integer-factor spline upsampling (order 3 = cubic), range-clipped |
| `crop_and_upsample` | function | Full-resolution crop plus upsample to an exact output size |
| `plane_to_tensor_px` | function | Exact float forward transform |
| `tensor_to_plane_px` | function | Exact float inverse transform |

## Inputs and outputs

`crop_plane(bands, center_px, output_size)` returns
`Ok((cropped, crop_origin))` where `cropped` is `(C, H_out, W_out)` and `crop_origin`
is the `(x, y)` full-plane pixel of its top-left corner; or `Err(FRAME_MALFORMED)`.

`decimate_to_size(bands, output_size)` returns `Ok((tensor, RoiTransform))` with
`scale_factor = 1 / factor`; `crop_and_upsample(bands, center_px, output_size,
upsample_factor, order)` returns `Ok((tensor, RoiTransform))` with
`scale_factor = upsample_factor`.

`plane_to_tensor_px` applies `tensor = (plane - crop_origin) * scale_factor`;
`tensor_to_plane_px` applies the exact inverse.

## Behavior

1. `crop_plane` computes the top-left corner from the centre and output size, clamps
   the window to the plane, and returns the actual origin.
2. `decimate_area` reshapes each plane into `factor x factor` blocks and takes the box
   mean, attenuating frequencies above the new Nyquist limit.
3. `decimate_to_size` picks `factor = min(H // H_out, W // W_out)`, centre-crops to
   `factor * output_size`, then decimates.
4. `upsample` uses `scipy.ndimage.zoom` with `grid_mode=True` and `reflect` boundary
   handling, then clips the result to the input `[min, max]` range. It returns
   `Err(FRAME_MALFORMED)` when `factor < 1` or `order` is outside `[0, 5]`.
5. `crop_and_upsample` crops `output_size / upsample_factor` at full resolution around
   `center_px` and upsamples to `output_size`.

## Errors and faults

`crop_plane`, `decimate_area`, `decimate_to_size`, `upsample`, and
`crop_and_upsample` return `Err(FaultCode.FRAME_MALFORMED)` on non-rank-3 input,
non-positive or non-fitting sizes, non-integer decimation, or an invalid upsample
factor or spline order.

## Messages

None.

## Configuration

Model input size comes from `InferenceConfig.input_height_px` and
`input_width_px` at the call site.

## Constraints

All functions are pure. The transform convention is linear,
`tensor_px = (plane_px - crop_origin_px) * scale_factor`; the half-pixel centre shift
of block averaging and zooming is ignored, which keeps the inverse exact.

## Related documents

- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.gimbal.pointing`](../gimbal/pointing.md)
