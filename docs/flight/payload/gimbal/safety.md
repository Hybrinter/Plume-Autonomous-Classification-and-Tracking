# flight.payload.gimbal.safety

**Source:** `packages/flight/src/flight/payload/gimbal/safety.py`
**Kind:** pure module

## Purpose

This module holds pre-arbiter safety gates. `PayloadController` applies them before
the arbiter step: confidence filter, minimum area filter, and deadband check.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `apply_confidence_gate` | function | Drops blobs below mean confidence threshold |
| `apply_min_area_gate` | function | Drops blobs below minimum pixel area |
| `check_deadband` | function | Classifies displacement against min and max deadband |
| `check_rate_limit` | function | Returns true when enough time elapsed for a new command |

## Inputs and outputs

Gates take blob tuples and thresholds; they return filtered blob tuples.

`check_deadband(displacement_px, min_px, max_px)` returns `Result[bool, FaultCode]`.

`check_rate_limit(last_command_time, now, rate_limit_hz)` returns bool.

## Behavior

1. `apply_confidence_gate` keeps blobs with `mean_confidence >= threshold`.
2. `apply_min_area_gate` keeps blobs with `pixel_area >= min_px`.
3. `check_deadband` returns `Ok(False)` below minimum (no command), `Ok(True)` in range
   (command allowed), and `Err(GIMBAL_RUNAWAY)` above maximum.
4. `check_rate_limit` returns true when `(now - last_command_time) >= 1/rate_limit_hz`.

## Errors and faults

| Result | Trigger |
| --- | --- |
| `Err(GIMBAL_RUNAWAY)` | Displacement exceeds `max_deadband_px` |

Deadband max violations increment strikes in `PayloadController` before escalation.

## Messages

None.

## Configuration

Uses `ControllerConfig.confidence_gate`, `min_blob_area_px`, `min_deadband_px`,
`max_deadband_px`, and `retarget_rate_limit_hz`.

## Constraints

STOW and ABSOLUTE requests from the arbiter are never suppressed by deadband logic in
the controller. Rate limiting for arbiter commands uses inline checks in the arbiter;
`check_rate_limit` is available for shared rate-limit math.

## Related documents

- [`flight.payload.gimbal.arbiter`](arbiter.md)
- [`flight.payload.control`](../control.md)
