# flight.payload.gimbal.request

**Source:** `packages/flight/src/flight/payload/gimbal/request.py`
**Kind:** pure module

## Purpose

`GimbalRequest` is the typed pose-command output from pure control cores. The payload
app shell maps it onto HAL pose calls and publishes a `GimbalCommandMsg` telemetry
record. Tracking torque is not a request.

The module also holds the mode-free control references `RateReference`,
`PoseReference`, `StowReference`, and `InhibitReference`, each carrying explicit
travel limits or a reason, plus the `validate_reference` check. These types are
inert foundations: the current app and controller still use `GimbalRequest` and
the scalar rate path, and the references are not wired into execution until the
runtime cutover.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `GimbalRequest` | dataclass | One pose command with mode, elevation, and reason |
| `TravelEnvelope` | dataclass | Elevation bounds and rate cap for a reference |
| `RateReference` | dataclass | Absolute rate plus travel envelope |
| `PoseReference` | dataclass | Pose target plus travel envelope |
| `StowReference` | dataclass | Stow target, travel envelope, and timeout |
| `InhibitReference` | dataclass | Actuator inhibit with a reason |
| `ControlReference` | union | Closed union of the four reference records |
| `validate_reference` | function | `Result[None, FaultCode]` validation of a reference |

## Inputs and outputs

`GimbalRequest` fields: `mode` (`GimbalCommandMode`), `el_deg`, `reason` (str).
Mode is `ABSOLUTE`, `STOW`, or `HOME`. `el_deg` is the target elevation for
`ABSOLUTE`. STOW and HOME ignore the value at the HAL after the shell maps the mode.

Reference fields: `TravelEnvelope` holds `theta_min_rad`, `theta_max_rad`, and
`omega_max_rad_s`. `RateReference` holds `rate_rad_s` and an envelope.
`PoseReference` and `StowReference` hold `target_rad` and an envelope; stow adds
`timeout_s`. `InhibitReference` holds `reason`.

`validate_reference` returns `Ok(None)` for a valid reference and
`Err(COMMAND_INVALID)` for a recoverable violation.

## Behavior

Pure cores construct a `GimbalRequest` and return it by value. The app shell selects
the HAL method from `mode` and publishes bus telemetry after the pose call.

`validate_reference` checks finite ordered envelope bounds, a positive rate cap,
finite rates (which may exceed the cap), pose and stow targets inside the
envelope inclusive, a finite positive stow timeout, and a nonempty inhibit
reason. Construction never raises for a recoverable validation failure.

## Errors and faults

`Err(COMMAND_INVALID)` covers every validation failure.

## Messages

None. The type is not a bus message. The shell publishes `GimbalCommandMsg` after
mapping to the HAL.

## Configuration

None.

## Constraints

Pure cores never publish to the bus or call the HAL. `GimbalRequest` flows only as a
return value. There is no azimuth field. Reference records carry explicit
numeric envelopes, not mode or graph names.

## Related documents

- [`flight.payload.gimbal.arbiter`](arbiter.md)
- [`flight.payload.app`](../app.md)
