# flight.payload.preprocess.crop

**Source:** `packages/flight/src/flight/payload/preprocess/crop.py`
**Kind:** pure module

## Purpose

This module crops a multispectral array to a fixed ROI and back-projects tensor pixels to
full-plane coordinates.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `crop_to_roi` | function | Crops `(C, H, W)` around a center pixel |
| `backproject_pixel` | function | Maps tensor pixel to full-frame pixel |

## Inputs and outputs

`crop_to_roi(bands, center_px, output_size)` returns `(cropped_bands, crop_origin)`.
`output_size` is `(H_out, W_out)`.

`backproject_pixel(px, crop_origin, scale_factor)` returns `(full_x, full_y)`.

## Behavior

1. Compute top-left `(x0, y0)` as center minus half the output size.
2. Clamp the window inside the image bounds.
3. Slice `bands[:, y0:y1, x0:x1]` and return the actual top-left as `crop_origin`.
4. Back-projection: `full = crop_origin + round(px / scale_factor)`.

When `scale_factor` is 1.0, back-projection is a translation by `crop_origin`.

## Errors and faults

None.

## Messages

None. Crop origin and scale factor copy into `InferenceResultMsg` for pointing math.

## Configuration

Output size comes from `InferenceConfig.input_height_px` and `input_width_px`.

## Constraints

Pure module. The app uses full-res crop in TRACKING and uniform decimation in search mode.

## Related documents

- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.gimbal.pointing`](gimbal/pointing.md)
- [`flight.payload.app`](app.md)
