# flight.payload.preprocess

**Source:** `packages/flight/src/flight/payload/preprocess`
**Kind:** package

## Purpose

The preprocess package holds pure functions that transform a prism camera buffer into
an inference-ready NCHW tensor. Stages run in a fixed order inside
`PayloadApp.process_frame` with no I/O and no global state.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`stack`](preprocess/stack.md) | module | Stack a prism buffer to `(3, H, W)` |
| [`radiometric`](preprocess/radiometric.md) | module | Per-channel dark, flat, and bad-pixel correction |
| [`normalize`](preprocess/normalize.md) | module | DN to [0, 1] scaling by ADC full scale |
| [`band_select`](preprocess/band_select.md) | module | Reorder band planes for model input |
| [`quality`](preprocess/quality.md) | module | Per-frame usability flags |

## Package interface

Re-exports: `MosaicCalibration`, `SmearRateSource`, `calibrate_mosaic`,
`compute_quality_flags`, `correct_bad_pixels`, `normalize_dn`, `select_bands`,
`stack_channels`.

## Interactions

None. Callers invoke these functions directly from the payload app. The published
tensor is local `(1, C, H, W)` float32. Quality flags attach to the in-process
processed frame record.

## Constraints

All functions are pure. The camera buffer is stacked once. Calibration runs per
channel with a one-pixel neighborhood. Quality flags run on the selected channels.
The published tensor is the full frame with a leading batch axis, with no crop and
no scale. The package does not read TOML or load files; artifact loading lives in
`flight.payload.calibration_io`.

## Related documents

- [`flight.payload.app`](app.md)
- [`flight.payload.calibration_io`](calibration_io.md)
