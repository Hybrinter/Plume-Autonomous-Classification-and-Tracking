# flight.payload.preprocess.normalize

**Source:** `packages/flight/src/flight/payload/preprocess/normalize.py`
**Kind:** pure module

## Purpose

This module scales calibrated band planes from digital numbers to the [0, 1] float32
domain. The model input contract and quality thresholds assume this range.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `normalize_dn` | function | Divides by ADC full scale and clips to [0, 1] |

## Inputs and outputs

`normalize_dn(planes, bit_depth)` takes `(C, H, W)` calibrated DN values and returns
`(C, H, W)` float32 in [0, 1].

## Behavior

1. Compute full scale as `2**bit_depth - 1`.
2. Divide each element by full scale.
3. Clip to [0, 1] and cast to float32.

## Errors and faults

None.

## Messages

None.

## Configuration

Uses `SensorConfig.bit_depth` (default 12, full scale 4095).

## Constraints

The function is pure with no I/O. Saturation detection downstream treats values at 1.0
as saturated after clipping.

## Related documents

- [`flight.payload.preprocess`](preprocess.md)
- [`flight.payload.preprocess.quality`](quality.md)
