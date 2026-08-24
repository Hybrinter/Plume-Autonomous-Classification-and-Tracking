# flight.payload.gimbal.safety

**Source:** `packages/flight/src/flight/payload/gimbal/safety.py`
**Kind:** pure module

## Purpose

Safety gates filter blobs and check displacement before the arbiter runs. All functions are pure.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `apply_confidence_gate` | function | Drops blobs below mean confidence |
| `apply_min_area_gate` | function | Drops blobs below pixel area |
| `check_deadband` | function | Tests displacement against min and max thresholds |
| `check_rate_limit` | function | Tests elapsed time since last command |

## Inputs and outputs

`apply_confidence_gate(blobs, threshold)` returns a filtered blob tuple.

`apply_min_area_gate(blobs, min_px)` returns a filtered blob tuple.

`check_deadband(displacement_px, min_px, max_px)` returns `Result[bool, FaultCode]`:
`Ok(False)` below min, `Ok(True)` in range, `Err(GIMBAL_RUNAWAY)` above max.

`check_rate_limit(last_command_time, now, rate_limit_hz)` returns a bool.

## Behavior

1. Confidence gate keeps blobs with `mean_confidence >= threshold`.
2. Area gate keeps blobs with `pixel_area >= min_px`.
3. Deadband below min returns no command. In range permits a command. Above max returns a fault.
4. Rate limit returns true when `(now - last_command_time) >= 1 / rate_limit_hz`.

The control core applies confidence and area gates before blob matching. It applies deadband
before arbiter refinement. The arbiter uses its own `_rate_ok` helper for command spacing.

## Errors and faults

| Result | Trigger |
| --- | --- |
| `Err(GIMBAL_RUNAWAY)` | Displacement exceeds `max_deadband_px` |

## Messages

None.

## Configuration

Reads `ControllerConfig`: `confidence_gate`, `min_blob_area_px`, `min_deadband_px`,
`max_deadband_px`, `retarget_rate_limit_hz`.

## Constraints

Pure module. STOW and ABSOLUTE requests are not deadband-suppressed in the control core.

## Related documents

- [`flight.payload.gimbal`](gimbal.md)
- [`flight.payload.gimbal.arbiter`](arbiter.md)
- [`flight.payload.control`](control.md)
