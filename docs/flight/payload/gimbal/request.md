# flight.payload.gimbal.request

**Source:** `packages/flight/src/flight/payload/gimbal/request.py`
**Kind:** pure module

## Purpose

`GimbalRequest` is the typed command output from the pure control core. The app shell maps it
to HAL calls and publishes telemetry.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `GimbalRequest` | class | Frozen command: mode, az, el, reason |

## Inputs and outputs

Fields: `mode` (`GimbalCommandMode`), `az_deg`, `el_deg`, `reason` (str).

RATE mode uses az and el as deg/s. ABSOLUTE mode uses deg. STOW and HOME ignore axis values.

## Behavior

The control core and arbiter return `GimbalRequest` by value. The app shell dispatches:

- RATE → `gimbal.set_rate`
- ABSOLUTE → `gimbal.goto_angle`
- STOW → `gimbal.stow`
- HOME → `gimbal.home`

## Errors and faults

None in this module. HAL errors surface in the app shell.

## Messages

None. The shell publishes `GimbalCommandMsg` after actuation.

## Configuration

None.

## Constraints

`GimbalRequest` never travels on the bus. The pure core has no HAL or bus access.

## Related documents

- [`flight.payload.gimbal`](gimbal.md)
- [`flight.payload.gimbal.arbiter`](arbiter.md)
- [`flight.payload.app`](app.md)
