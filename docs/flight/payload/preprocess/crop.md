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

Uses `InferenceConfig.input_height_px` and `input_width_px` for ROI size when a caller
requests a crop.

## Constraints

All functions are pure. The live payload app does not call this module. It passes the
full band plane with crop origin `(0, 0)` and scale 1. Tests keep the library.

## Related documents

- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.gimbal.pointing`](../gimbal/pointing.md)
