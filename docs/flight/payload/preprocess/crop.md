# flight.payload.preprocess.crop

**Source:** `packages/flight/src/flight/payload/preprocess/crop.py`
**Kind:** pure module

## Purpose

This module crops a multispectral array to a fixed ROI and back-projects pixel
coordinates from the cropped tensor to full band-plane space. Pointing math uses the
crop origin and scale factor from preprocessing.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `crop_to_roi` | function | Crops `(C, H, W)` bands around a center pixel |
| `backproject_pixel` | function | Maps cropped tensor pixels to full-frame pixels |

## Inputs and outputs

`crop_to_roi(bands, center_px, output_size)` returns
`(cropped_bands, crop_origin)` where `cropped_bands` is `(C, H_out, W_out)` and
`crop_origin` is `(x, y)` in full-plane pixels.

`backproject_pixel(px, crop_origin, scale_factor)` returns `(x, y)` in full-frame
space.

## Behavior

1. `crop_to_roi` computes the top-left corner from the center and output size.
2. It clamps the window to image bounds when the centered window would extend outside.
3. It slices the band array and returns the actual origin (which may differ after
   clamping).
4. `backproject_pixel` applies `full = crop_origin + round(px / scale_factor)`.

## Errors and faults

None.

## Messages

None.

## Configuration

Uses `InferenceConfig.input_height_px` and `input_width_px` for ROI size in the
payload app.

## Constraints

All functions are pure. In search mode the app decimates the full plane and sets
`crop_origin=(0,0)` with `scale_factor=1/factor`. In TRACKING mode it calls
`crop_to_roi` at full resolution with `scale_factor=1.0`.

## Related documents

- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.gimbal.pointing`](../gimbal/pointing.md)
