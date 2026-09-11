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
| [`quality`](preprocess/quality.md) | module | Per-frame usability flags and keep/drop verdict |
| [`crop`](preprocess/crop.md) | module | ROI crop, decimate/upsample resampling, and pixel back-projection |

## Package interface

Re-exports: `CELL_OFFSETS`, `DEFAULT_BAND_NAMES`, `DEFAULT_USABILITY_POLICY`,
`SATURATION_PIXEL_LEVEL`, `CloudTest`, `MosaicCalibration`, `QualityMetrics`,
`RoiTransform`, `UsabilityPolicy`, `backproject_pixel`, `band_index`,
`calibrate_mosaic`, `cloud_fraction`, `compute_quality_flags`,
`compute_quality_metrics`, `correct_bad_pixels`, `crop_and_upsample`, `crop_plane`,
`crop_to_roi`, `dark_matches_frame`, `decide_usability`, `decimate_area`,
`decimate_to_size`, `flags_from_metrics`, `full_scale_dn`, `interleave_bands`,
`normalize_dn`, `plane_to_tensor_px`, `predicted_smear_px`,
`saturated_fraction_normalized`, `saturated_fraction_raw`,
`scale_to_reference_exposure`, `select_bands`, `separate_bands`,
`tensor_to_plane_px`, `upsample`.

## Interactions

None. Callers invoke these functions directly from the payload app. Outputs feed the
detector as a local `(C, H, W)` float32 array and quality flags attached to the
in-process processed frame record.

## Constraints

All functions are pure. Calibration runs on the raw mosaic plane before CFA separation.
Quality flags run on the full band plane. The full selected band plane is passed to
inference with no crop and no scale. The package does not read TOML or load files;
artifact loading lives in `flight.payload.calibration_io`.

## Related documents

- [`flight.payload.app`](app.md)
- [`flight.payload.calibration_io`](calibration_io.md)
