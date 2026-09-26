# flight.payload.preprocess

**Source:** `packages/flight/src/flight/payload/preprocess`
**Kind:** package

## Purpose

The preprocess package holds pure functions that transform a raw RGB stack into an
inference-ready tensor. Stages run in a fixed order inside `PayloadApp.process_frame`
with no I/O and no global state.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`radiometric`](preprocess/radiometric.md) | module | Per-channel dark, flat, and bad-pixel correction |
| [`demosaic`](preprocess/demosaic.md) | module | Confirms the stack matches `BAND_ORDER` |
| [`normalize`](preprocess/normalize.md) | module | DN to [0, 1] scaling by ADC full scale |
| [`band_select`](preprocess/band_select.md) | module | Publishes the `BAND_ORDER` names |
| [`quality`](preprocess/quality.md) | module | Per-frame usability flags |
| [`resample`](preprocess/resample.md) | module | Cubic or linear upscale after plane confirm |

## Package interface

Re-exports: `MosaicCalibration`, `SmearRateSource`, `calibrate_mosaic`,
`canonical_band_names`, `compute_quality_flags`, `confirm_planes`,
`correct_bad_pixels`, `dark_matches_frame`, `normalize_dn`, `upsample_planes`.

## Interactions

None. Callers invoke these functions directly from the payload app. Outputs feed the
detector as a local `(3, H, W)` float32 array and quality flags attached to the
in-process processed frame record.

## Constraints

All functions are pure. Channel order is `BAND_ORDER` (BLUE, GREEN, RED). The
package does not reorder planes. Quality flags run on the native normalized
stack. Upsample runs after a clean quality result. The package does not read
TOML or load files. Artifact loading lives in `flight.payload.calibration_io`.

## Related documents

- [`flight.payload.app`](app.md)
- [`flight.payload.calibration_io`](calibration_io.md)
