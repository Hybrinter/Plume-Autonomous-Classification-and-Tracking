# flight.payload.control

**Source:** `packages/flight/src/flight/payload/control.py`
**Kind:** pure module

## Purpose

`PayloadController` is the pure payload control core. It composes blob gates, tracking
estimators, deadband and runaway checks, the gimbal arbiter, and the LQR law into one
`step` call. All state lives in `ControlState` and is replaced each frame.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ControlState` | dataclass | Bundled arbiter, EMA, Kalman, runaway, deadband, and rate state |
| `PayloadController` | dataclass | Immutable control core with config and sub-components |
| `PayloadController.from_config` | static method | Builds arbiter, Kalman, LQR, and plane geometry from config |
| `PayloadController.initial_state` | method | Returns IDLE arbiter with uninitialized EMA |
| `PayloadController.step` | method | Runs one pure control step |

## Inputs and outputs

`from_config` takes `ControllerConfig` and `SensorConfig`. It returns a
`PayloadController`.

`step` takes `ControlState`, `InferenceResultMsg`, monotonic `now`, optional
`GimbalPosition`, and SAFE flags. It returns
`(ControlState, GimbalRequest | None, list[TelemetryEventMsg], FaultCode | None)`.

## Behavior

1. Apply confidence and minimum-area gates to inference blobs.
2. Match blobs to the previous frame by IoU and assign persistence counts.
3. When a match exists, compute boresight error in degrees and full-plane pixel
   displacement from the blob centroid.
4. Update the EMA with boresight error, or reset it when no match exists.
5. Predict the Kalman state, then update with the EMA observation when initialized.
6. Run the deadband check on displacement. Suppress RATE commands below the minimum.
   Count strikes above the maximum.
7. Step the gimbal arbiter with filtered blobs and boresight error.
8. When the arbiter issues a RATE request and the EMA is initialized, replace rates with
   LQR output (`-K @ x`), clamped to the max slew rate.
9. Suppress RATE requests when the deadband gate blocks them.
10. Compare commanded rates to measured encoder motion via the runaway monitor.
11. Thread commanded az/el rates into the next state for the runaway check.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `GIMBAL_RUNAWAY` | Deadband strikes reach `max_deadband_strike_count`, or encoder divergence reaches `runaway_strike_count` |

STOW and ABSOLUTE requests from the arbiter are never suppressed by the deadband gate.

## Messages

None. The pure core returns `GimbalRequest` and `TelemetryEventMsg` values; the app
shell publishes them.

## Configuration

Reads `ControllerConfig` (gates, persistence, Kalman, LQR, deadband, runaway, slew
limits) and band-plane geometry from `SensorConfig` (`width_px // 2`,
`height_px // 2`, `ifov_deg_per_px`).

## Constraints

The module performs no I/O, bus access, or clock reads. Time arrives only through the
`now` argument. State is immutable: each step returns a new `ControlState`. Boresight
error and Kalman state use degree space; the LQR setpoint is zero error.

## Related documents

- [`flight.payload.gimbal`](gimbal.md)
- [`flight.payload.tracking`](tracking.md)
- [`flight.payload.app`](app.md)
