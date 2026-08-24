# flight.payload.preprocess.normalize

**Source:** `packages/flight/src/flight/payload/preprocess/normalize.py`
**Kind:** pure module

## Purpose

This module scales calibrated DN band planes to the `[0, 1]` float32 domain expected by quality
checks and model input.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `normalize_dn` | function | Divides by ADC full scale and clips |

## Inputs and outputs

`normalize_dn(planes, bit_depth)` takes `(C, H, W)` calibrated values. It returns `(C, H, W)`
float32 in `[0, 1]`.

## Behavior

1. Compute full scale as `2**bit_depth - 1`.
2. Divide each element by full scale.
3. Clip to `[0, 1]` and cast to float32.

Values below zero clip to 0.0. Values above full scale clip to 1.0.

## Errors and faults

None.

## Messages

None.

## Configuration

Reads `SensorConfig.bit_depth`.

## Constraints

Pure module.

## Related documents

- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.preprocess.radiometric`](radiometric.md)
