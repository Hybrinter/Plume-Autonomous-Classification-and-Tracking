# flight.payload.gimbal.safety

**Source:** `packages/flight/src/flight/payload/gimbal/safety.py`
**Kind:** pure module

## Purpose

This module holds pre-arbiter safety gates. `PayloadController` applies them before
blob matching: confidence filter and minimum area filter.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `apply_confidence_gate` | function | Drops blobs below mean confidence threshold |
| `apply_min_area_gate` | function | Drops blobs below minimum pixel area |

## Inputs and outputs

Gates take blob tuples and thresholds; they return filtered blob tuples.

## Behavior

1. `apply_confidence_gate` keeps blobs with `mean_confidence >= threshold`.
2. `apply_min_area_gate` keeps blobs with `pixel_area >= min_px`.

## Errors and faults

None.

## Messages

None.

## Configuration

Uses `VisionConfig.confidence_gate` and `min_blob_area_px`.

## Constraints

The functions are pure. They do not rate-limit gimbal commands.

## Related documents

- [`flight.payload.gimbal.arbiter`](arbiter.md)
- [`flight.payload.control`](../control.md)
