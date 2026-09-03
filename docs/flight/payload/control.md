# flight.payload.control

**Source:** `packages/flight/src/flight/payload/control.py`
**Kind:** pure module

## Purpose

`PayloadController` is the pure payload control core. It composes blob gates, tracking
estimators, encoder runaway, the gimbal arbiter, and the LQR law into one `step` call.
All state lives in `ControlState` and is replaced each frame.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ControlState` | dataclass | Bundled arbiter, EMA, Kalman, runaway, and commanded-rate state |
| `PayloadController` | dataclass | Immutable control core with config and sub-components |
| `PayloadController.from_config` | static method | Builds arbiter, Kalman, LQR, and plane geometry from config |
| `PayloadController.initial_state` | method | Returns IDLE arbiter with uninitialized EMA |
| `PayloadController.step` | method | Runs one pure control step |

## Inputs and outputs

`from_config` takes `ControllerConfig`, `SensorConfig`, and `GimbalConfig`. It returns a
`PayloadController`.

`step` takes `ControlState`, `InferenceResultMsg`, monotonic `now`, optional
`GimbalPosition`, and SAFE flags. It returns
`(ControlState, GimbalRequest | None, list[TelemetryEventMsg], FaultCode | None)`.

## Behavior

1. Apply confidence and minimum-area gates to inference blobs.
2. Match blobs to the previous frame by IoU and assign persistence counts.
3. When a match exists, compute boresight error in degrees from the blob centroid.
4. Update the EMA with boresight error, or reset it when no match exists.
5. Predict the Kalman state, then update with the EMA observation when initialized.
6. Step the gimbal arbiter with filtered blobs, boresight error, and current elevation.
7. When the arbiter issues a RATE request and the EMA is initialized, replace rates with
   LQR output (`-K @ x`), clamped to the hardware slew cap.
8. Compare commanded rates to measured encoder motion via the runaway monitor.
9. Thread commanded az/el rates into the next state for the runaway check.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `GIMBAL_RUNAWAY` | Encoder divergence reaches `runaway_strike_count` |

## Messages

None. The pure core returns `GimbalRequest` and `TelemetryEventMsg` values; the app
shell publishes them.

## Configuration

Reads `ControllerConfig` (gates, persistence, Kalman, LQR, runaway) and band-plane
geometry from `SensorConfig` (`width_px // 2`, `height_px // 2`,
`ifov_band_deg_per_px`). Hardware slew clamp comes from `GimbalConfig`.

## Constraints

The module performs no I/O, bus access, or clock reads. Time arrives only through the
`now` argument. State is immutable: each step returns a new `ControlState`. Boresight
error and Kalman state use degree space; the LQR setpoint is zero error.

## Related documents

- [`flight.payload.gimbal`](gimbal.md)
- [`flight.payload.tracking`](tracking.md)
- [`flight.payload.app`](app.md)
