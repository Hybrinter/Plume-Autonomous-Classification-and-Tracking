# flight.payload.gimbal.request

**Source:** `packages/flight/src/flight/payload/gimbal/request.py`
**Kind:** pure module

## Purpose

`GimbalRequest` is the typed command output from pure control cores. The payload app
shell maps it onto HAL calls and publishes a `GimbalCommandMsg` telemetry record.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `GimbalRequest` | dataclass | One gimbal command with mode, axis values, and reason |

## Inputs and outputs

Fields: `mode` (`GimbalCommandMode`), `az_deg`, `el_deg`, `reason` (str).

For RATE mode, az and el are degrees per second. For ABSOLUTE mode, they are target
angles in degrees. STOW and HOME ignore axis values.

## Behavior

Pure cores construct a `GimbalRequest` and return it by value. The app shell selects the
HAL method from `mode` and publishes bus telemetry after actuation.

## Errors and faults

None.

## Messages

None. The type is not a bus message. The shell publishes `GimbalCommandMsg` after
mapping to the HAL.

## Configuration

None.

## Constraints

Pure cores never publish to the bus or call the HAL. `GimbalRequest` flows only as a
return value from `PayloadController.step` or `GimbalArbiter.step`.

## Related documents

- [`flight.payload.gimbal.arbiter`](arbiter.md)
- [`flight.payload.app`](../app.md)
