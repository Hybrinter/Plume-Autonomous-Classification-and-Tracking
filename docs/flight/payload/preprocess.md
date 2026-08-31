# flight.payload.preprocess

**Source:** `packages/flight/src/flight/payload/preprocess`
**Kind:** package

## Purpose

The preprocess package holds pure functions that transform a raw mosaic plane into an
inference-ready tensor. Stages run in a fixed order inside `PayloadApp.process_frame`
with no I/O and no global state.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`radiometric`](preprocess/radiometric.md) | module | Mosaic-plane dark, flat, and bad-pixel correction |
| [`demosaic`](preprocess/demosaic.md) | module | 2x2 CFA separation and interleave |
| [`normalize`](preprocess/normalize.md) | module | DN to [0, 1] scaling by ADC full scale |
| [`band_select`](preprocess/band_select.md) | module | Reorder band planes for model input |
| [`quality`](preprocess/quality.md) | module | Per-frame usability flags |
| [`crop`](preprocess/crop.md) | module | ROI crop and pixel back-projection |

## Package interface

Re-exports: `CELL_OFFSETS`, `MosaicCalibration`, `backproject_pixel`, `calibrate_mosaic`,
`compute_quality_flags`, `correct_bad_pixels`, `crop_to_roi`, `interleave_bands`,
`normalize_dn`, `select_bands`, `separate_bands`.

## Interactions

None. Callers invoke these functions directly from the payload app. Outputs feed the
detector as a local `(C, H, W)` float32 array and quality flags attached to the
in-process processed frame record.

## Constraints

All functions are pure. Calibration runs on the raw mosaic plane before CFA separation.
Quality flags run on the full band plane. The live app does not crop or decimate. The package does
not read TOML or load files; artifact loading lives in `flight.payload.calibration_io`.

## Related documents

- [`flight.payload.app`](app.md)
- [`flight.payload.calibration_io`](calibration_io.md)
