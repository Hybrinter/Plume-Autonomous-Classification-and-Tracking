# flight.payload.calibration_io

**Source:** `packages/flight/src/flight/payload/calibration_io.py`
**Kind:** module

## Purpose

This module loads mosaic calibration artifacts at startup. It reads checksummed `.npy`
files from a calibration directory and returns a `MosaicCalibration`. It also builds an
identity calibration for SIL and development when no artifact directory is configured.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `build_identity_calibration` | function | Returns zero dark, unit flat, empty bad-pixel mask |
| `load_calibration` | function | Loads and verifies dark, flat, and bad-pixel artifacts |

## Inputs and outputs

`build_identity_calibration(height_px, width_px)` returns a `MosaicCalibration` with
shape `(height_px, width_px)`.

`load_calibration(calibration_dir, height_px, width_px)` returns
`Result[MosaicCalibration, FaultCode]`.

## Behavior

1. `build_identity_calibration` allocates zero dark frame, unit flat field, and an all-false
   bad-pixel mask.
2. `load_calibration` reads `manifest.json` in the calibration directory.
3. For each required artifact (`dark_frame`, `flat_field`, `bad_pixel_mask`), it reads
   the named `.npy` file, verifies the SHA-256 digest against the manifest, and loads
   the array.
4. It checks that each array shape matches `(height_px, width_px)`.
5. On success it returns a `MosaicCalibration` with float32 dark and flat arrays and a
   bool bad-pixel mask.

## Errors and faults

| Result | Trigger |
| --- | --- |
| `Err(CALIBRATION_INVALID)` | Missing directory, unreadable manifest, missing artifact entry, digest mismatch, unreadable `.npy`, wrong shape, or malformed JSON |

The composition root treats this error as an unrecoverable startup failure.

## Messages

None.

## Configuration

Uses `SensorConfig.calibration_dir`, `height_px`, and `width_px`. An empty
`calibration_dir` selects `build_identity_calibration` at the composition root.

## Constraints

This module performs file I/O. The preprocess package stays pure and does not load
artifacts. Flight deployments must supply real characterization artifacts; identity
calibration is for SIL and development only.

## Related documents

- [`flight.payload.preprocess.radiometric`](preprocess/radiometric.md)
- [`flight.payload.preprocess`](preprocess.md)
