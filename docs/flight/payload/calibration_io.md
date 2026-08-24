# flight.payload.calibration_io

**Source:** `packages/flight/src/flight/payload/calibration_io.py`
**Kind:** module

## Purpose

This module loads mosaic calibration artifacts at startup. It reads checksummed `.npy` files and
returns a `MosaicCalibration` for the composition root.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `build_identity_calibration` | function | Builds zero dark, unit flat, no bad pixels |
| `load_calibration` | function | Loads and verifies artifacts from a directory |

## Inputs and outputs

`build_identity_calibration(height_px, width_px)` returns a `MosaicCalibration` with shape
`(height_px, width_px)`.

`load_calibration(calibration_dir, height_px, width_px)` returns
`Result[MosaicCalibration, FaultCode]`.

## Behavior

1. `load_calibration` reads `manifest.json` in `calibration_dir`.
2. For each of `dark_frame`, `flat_field`, and `bad_pixel_mask`, it reads the listed `.npy`
   file, verifies SHA-256 against the manifest, and loads the array.
3. It checks each artifact shape against `(height_px, width_px)`.
4. On success it casts dark and flat to float32 and the mask to bool.
5. `build_identity_calibration` returns zeros, ones, and an all-false mask without file I/O.

## Errors and faults

| Result | Trigger |
| --- | --- |
| `Err(CALIBRATION_INVALID)` | Missing directory, bad manifest, missing file, checksum mismatch, bad shape, unreadable `.npy` |

The composition root treats startup load failure as unrecoverable.

## Messages

None.

## Configuration

Uses `SensorConfig.height_px`, `width_px`, and `calibration_dir`. An empty `calibration_dir`
selects identity calibration in the composition root.

## Constraints

This module performs file I/O. The preprocess package stays pure. Identity calibration is for
SIL and development when `calibration_dir` is empty.

## Related documents

- [`flight.payload`](payload.md)
- [`flight.payload.preprocess.radiometric`](preprocess/radiometric.md)
