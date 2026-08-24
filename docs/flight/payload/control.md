# flight.payload.control

**Source:** `packages/flight/src/flight/payload/control.py`
**Kind:** pure module

## Purpose

The payload control core composes tracking estimators and the gimbal FSM into one pure step. It
maps an inference result and prior state to a new state and an optional gimbal request.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ControlState` | class | Bundled arbiter, EMA, Kalman, runaway, deadband, and commanded-rate state |
| `PayloadController` | class | Immutable control core built from config |
| `PayloadController.from_config` | function | Builds arbiter, Kalman, LQR, and pointing geometry |
| `PayloadController.initial_state` | function | Returns IDLE arbiter and uninitialized estimators |
| `PayloadController.step` | function | Runs one control step |

## Inputs and outputs

`from_config(cfg, sensor)` returns a `PayloadController`. Band plane size is
`sensor.{width,height}_px // 2`.

`initial_state()` returns a `ControlState`.

`step(state, result, now, gimbal_pos, safe_commanded, safe_cleared)` returns
`(ControlState, GimbalRequest | None, list[TelemetryEventMsg], FaultCode | None)`.

## Behavior

1. Apply confidence and min-area gates, then `match_blobs` against prior tracked blobs.
2. When a blob matches, compute boresight error in degrees and full-plane displacement in
   pixels.
3. Update the EMA on error degrees; reset EMA when no match.
4. Run Kalman predict every frame; run update only when EMA is initialized.
5. Run the deadband gate on displacement. Max-deadband violations increment strikes and may
   suppress or fault.
6. Call `GimbalArbiter.step` with filtered blobs and error degrees.
7. Refine RATE requests with LQR when EMA is initialized: physical rate is `-u` from
   `compute_control`, clamped to `max_slew_rate_deg_per_s`.
8. Suppress RATE requests when the deadband gate blocks them. STOW and ABSOLUTE are not
   suppressed.
9. Run the encoder runaway monitor against prior commanded rates.
10. Store commanded RATE values in state for the next frame's runaway check.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `GIMBAL_RUNAWAY` | Deadband strikes reach `max_deadband_strike_count`, or encoder runaway strikes |

## Messages

None. The arbiter returns `TelemetryEventMsg` values; the app shell publishes them.

## Configuration

Reads `ControllerConfig` and `SensorConfig.ifov_deg_per_px`.

Fields include gates, persistence, Kalman and LQR tuning, slew limits, and runaway thresholds.

## Constraints

Pure module: no I/O, no bus access, no clock reads. State is frozen and replaced each step.
Estimators operate in boresight-error degree space.

## Related documents

- [`flight.payload`](payload.md)
- [`flight.payload.gimbal`](gimbal.md)
- [`flight.payload.tracking`](tracking.md)
