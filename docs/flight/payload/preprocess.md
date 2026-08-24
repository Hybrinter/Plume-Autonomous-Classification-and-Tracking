# flight.payload.preprocess

**Source:** `packages/flight/src/flight/payload/preprocess/`
**Kind:** package

## Purpose

The preprocess package holds pure functions that transform a raw mosaic plane into a normalized,
band-ordered tensor for inference.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`radiometric`](preprocess/radiometric.md) | module | Mosaic calibration and bad-pixel repair |
| [`demosaic`](preprocess/demosaic.md) | module | 2x2 CFA separation and interleave |
| [`normalize`](preprocess/normalize.md) | module | DN to [0, 1] scaling |
| [`band_select`](preprocess/band_select.md) | module | Reorder planes to model input bands |
| [`quality`](preprocess/quality.md) | module | Per-frame usability flags |
| [`crop`](preprocess/crop.md) | module | ROI crop and pixel back-projection |

## Package interface

Re-exports: `MosaicCalibration`, `calibrate_mosaic`, `correct_bad_pixels`, `separate_bands`,
`interleave_bands`, `CELL_OFFSETS`, `normalize_dn`, `select_bands`, `compute_quality_flags`,
`crop_to_roi`, `backproject_pixel`.

## Interactions

None. The app shell calls these functions inside `process_frame()`. Calibration artifacts load
through `flight.payload.calibration_io`.

## Constraints

All functions are pure with no I/O. Calibration runs on the raw mosaic before CFA separation.
Quality flags run on the full band plane before ROI selection.

## Related documents

- [`flight.payload`](payload.md)
- [`flight.payload.app`](app.md)
- [`flight.payload.calibration_io`](calibration_io.md)
