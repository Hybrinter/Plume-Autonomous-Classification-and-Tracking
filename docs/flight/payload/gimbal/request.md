# flight.payload.gimbal.request

**Source:** `packages/flight/src/flight/payload/gimbal/request.py`
**Kind:** pure module

## Purpose

The module holds the mode-free control references `RateReference`,
`PoseReference`, `StowReference`, and `InhibitReference`, each carrying
explicit travel limits or a reason, plus the `validate_reference` check.
Payload graphs emit references in their outcomes; the `ServoController` maps
them onto bounded rates and the app shell commits them onto `GimbalActuator`
HAL calls.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TravelEnvelope` | dataclass | Elevation bounds and rate cap for a reference |
| `RateReference` | dataclass | Absolute rate plus travel envelope |
| `PoseReference` | dataclass | Pose target plus travel envelope |
| `StowReference` | dataclass | Stow target, travel envelope, and timeout |
| `InhibitReference` | dataclass | Actuator inhibit with a reason |
| `ControlReference` | union | Closed union of the four reference records |
| `validate_reference` | function | `Result[None, FaultCode]` validation of a reference |

## Inputs and outputs

`TravelEnvelope` holds `theta_min_rad`, `theta_max_rad`, and
`omega_max_rad_s`. `RateReference` holds `rate_rad_s` and an envelope.
`PoseReference` and `StowReference` hold `target_rad` and an envelope; stow
adds `timeout_s`. `InhibitReference` holds `reason`.

`validate_reference` returns `Ok(None)` for a valid reference and
`Err(COMMAND_INVALID)` for a recoverable violation.

## Behavior

Pure graph outcomes construct a `ControlReference` and return it by value. The
app shell commits a changed reference once per outcome: it records it on
`PayloadState.reference`, arms any stow or goto HAL metadata for
`StowReference`/`PoseReference`, and publishes a `GimbalCommandMsg` audit
record. `ServoController.inner_step` consumes the committed reference each
inner tick.

`validate_reference` checks finite ordered envelope bounds, a positive rate
cap, finite rates (which may exceed the cap), pose and stow targets inside the
envelope inclusive, a finite positive stow timeout, and a nonempty inhibit
reason. Construction never raises for a recoverable validation failure.

## Errors and faults

`Err(COMMAND_INVALID)` covers every validation failure. The shell inhibits,
latches containment, and publishes a fault on an invalid committed reference;
a failed stow or goto HAL metadata call also latches containment and returns
the HAL fault without a successful audit.

## Messages

None. The type is not a bus message. The shell publishes `GimbalCommandMsg`
after a successful reference commit.

## Configuration

None.

## Constraints

Pure cores never publish to the bus or call the HAL. References flow only as
return values and committed state fields. There is no azimuth field. Reference
records carry explicit numeric envelopes, not mode or graph names.

## Related documents

- [`flight.payload.control`](../control.md)
- [`flight.payload.app`](../app.md)
- [`flight.payload.graphs.runtime`](../graphs/runtime.md)
