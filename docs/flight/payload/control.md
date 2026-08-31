# flight.payload.control

**Source:** `packages/flight/src/flight/payload/control.py`
**Kind:** pure module

## Purpose

`PayloadController` is the pure payload control core. It splits work into
`ingest_vision`, `step_outer`, and `step_inner`. `step` runs ingest then one outer tick
for the frame-synchronous app path.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ControlState` | dataclass | Arbiter, EMA, Kalman, servo, rewind ring, held `r`, and vision mailbox |
| `PayloadController` | dataclass | Immutable control core with config and sub-components |
| `PayloadController.from_config` | static method | Builds arbiter, Kalman, LQR, servo plant, and plane geometry |
| `PayloadController.initial_state` | method | Returns IDLE arbiter with uninitialized EMA and `r = 0` |
| `PayloadController.ingest_vision` | method | Gates blobs, forms CoM, writes the vision mailbox |
| `PayloadController.step_outer` | method | Predict, encoder update, rewind, LQR, arbiter |
| `PayloadController.step_inner` | method | Rate PI in SI units; returns torque in N·m |
| `PayloadController.step` | method | `ingest_vision` then one `step_outer` |

## Inputs and outputs

`from_config` takes `ControllerConfig`, `SensorConfig`, and optional `GimbalConfig`. It
returns a `PayloadController`.

`ingest_vision` takes `ControlState`, `InferenceResultMsg`, and shutter time. It returns
an updated `ControlState`.

`step_outer` takes `ControlState`, optional `GimbalPosition`, monotonic `now`, `dt`, and
SAFE flags. It returns
`(ControlState, GimbalRequest | None, list[TelemetryEventMsg], FaultCode | None)`.

`step_inner` takes `ControlState`, optional `GimbalPosition`, and `dt`. It returns
`(ControlState, (tau_az_nm, tau_el_nm))`.

`step` uses the same signature as the previous single-step controller and calls ingest
then outer.

## Behavior

1. `ingest_vision` applies confidence and minimum-area gates, then IoU matching.
2. It forms an area-weighted CoM of matched blobs and converts that CoM through IFOV
   to boresight-error degrees. Crop origin and scale come from the inference result.
3. It updates the EMA and writes `(z_az, z_el, t_shutter)` into the vision mailbox.
4. `step_outer` predicts both Kalman axes with the held `r`, chunked at
   `dt_outer_max_s`.
5. It updates from encoder `theta_g` when a position is present, then pushes a rewind
   snapshot.
6. It consumes the mailbox through the rewind ring once. Stale samples older than the
   ring are dropped. `r` stays 0 until the first accepted vision update.
7. It steps the arbiter with caller `dt`. Acquire and release counters advance only when
   a new frame is pending.
8. In TRACKING after the first vision sample, LQR overwrites `r`. In SCAN, a stub
   `r = scan_kp * (theta_cmd - theta_enc)` fills the same register. SAFE resets the PI
   integrator and issues STOW.
9. `step_inner` converts `r` and encoder angles to radians, chunks `dt` at
   `dt_inner_max_s`, and calls the rate servo.

## Errors and faults

The live path does not raise deadband or encoder-runaway faults. Kalman updates that
meet a singular innovation covariance are skipped. `step_outer` returns `fault=None`.

## Messages

None. The pure core returns `GimbalRequest` and `TelemetryEventMsg` values; the app
shell publishes them.

## Configuration

Reads `ControllerConfig` (gates, persistence, Kalman, LQR, servo gains, Δt bounds,
`scan_kp`) and band-plane geometry from `SensorConfig` (`width_px // 2`,
`height_px // 2`, `ifov_deg_per_px`). Inertia `J` and damping `B` come from
`GimbalConfig`.

## Constraints

The module performs no I/O, bus access, or clock reads. Time arrives only through `now`
and `dt`. State is immutable: each call returns a new `ControlState`. Pointing and LQR
use degrees. The inner servo uses radians and newton-metres.

## Related documents

- [`flight.payload.gimbal`](gimbal.md)
- [`flight.payload.tracking`](tracking.md)
- [`flight.payload.app`](app.md)
