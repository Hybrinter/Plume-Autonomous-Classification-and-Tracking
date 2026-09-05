# flight.payload.gimbal.request

**Source:** `packages/flight/src/flight/payload/gimbal/request.py`
**Kind:** pure module

## Purpose

`GimbalRequest` is the typed pose-command output from pure control cores. The payload
app shell maps it onto HAL pose calls and publishes a `GimbalCommandMsg` telemetry
record. Tracking torque is not a request.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `GimbalRequest` | dataclass | One pose command with mode, elevation, and reason |

## Inputs and outputs

Fields: `mode` (`GimbalCommandMode`), `el_deg`, `reason` (str).

Mode is `ABSOLUTE`, `STOW`, or `HOME`. `el_deg` is the target elevation for
`ABSOLUTE`. STOW and HOME ignore the value at the HAL after the shell maps the mode.

## Behavior

Pure cores construct a `GimbalRequest` and return it by value. The app shell selects
the HAL method from `mode` and publishes bus telemetry after the pose call.

## Errors and faults

None.

## Messages

None. The type is not a bus message. The shell publishes `GimbalCommandMsg` after
mapping to the HAL.

## Configuration

None.

## Constraints

Pure cores never publish to the bus or call the HAL. `GimbalRequest` flows only as a
return value. There is no azimuth field.

## Related documents

- [`flight.payload.gimbal.arbiter`](arbiter.md)
- [`flight.payload.app`](../app.md)
