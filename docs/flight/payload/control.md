# flight.payload.control

**Source:** `packages/flight/src/flight/payload/control.py`

**Kind:** pure module

## Purpose

`ServoController` is the pure mode-free servo core. It maps a typed
`ControlReference` produced by the active payload graph onto a bounded rate
reference, then advances the encoder ring, the polynomial rate estimate, and
the PI + computed-torque inner loop. It holds no graph, mode, or vision
knowledge; SAFE and inhibit reach it only as references.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `EncoderState` | dataclass | Encoder sample ring, last angle, and measured rate |
| `InnerControlState` | dataclass | Inner PI integrator, last inner time, and last torque |
| `IntegrityState` | dataclass | Encoder-freeze strike counter |
| `ServoState` | dataclass | The physical-plant memory threaded across inner ticks |
| `InnerTick` | dataclass | Updated `ServoState` plus the torque command |
| `ServoController` | dataclass | Immutable mode-free servo core |
| `ServoController.from_config` | static method | Builds the core from typed config slices |
| `ServoController.initial_state` | method | Empty encoder ring, zeroed PI and rate |
| `ServoController.reference_rate` | method | Maps a reference onto a bounded absolute rate |
| `ServoController.inner_step` | method | Pushes the encoder, fits the rate, emits torque |

## Inputs and outputs

`from_config` takes `ControllerConfig` and `GimbalConfig` slices (inner,
position, and integrity sections plus plant values). `reference_rate` takes a
`ControlReference`, the current encoder elevation in radians, and the
`detailed_plant` flag; it returns `Result[float, FaultCode]` -
`Err(COMMAND_INVALID)` on an invalid reference or nonfinite elevation, `0.0`
for an `InhibitReference`. `inner_step` takes the `ServoState`, `now`, one
`EncoderSample`, the committed reference, and an optional inner period; it
returns an `InnerTick` with the updated state and the commanded torque.

## Behavior

1. `reference_rate` validates the reference with `validate_reference`, then
   maps it: `InhibitReference` to zero, `PoseReference`/`StowReference` through
   `position_rate` clipped by the position cap and the envelope rate cap, and
   `RateReference` through the directional `stopping_cap` guard against the
   science-boundary guard offset. The unbounded (production) form of
   `stopping_cap` leaves outward rates at zero on an exhausted bound.
2. `inner_step` appends the sample to the bounded encoder ring
   (`rate_fit_n` entries), fits `measured_rate_rad_s` with
   `fit_rate_timed`, and runs the PI through `gimbal.inner_step`.
3. An `InhibitReference` or an invalid reference produces a zero tick: the
   encoder frame still updates but the integrator, torque, and commanded rate
   reset to zero.
4. At a hardware stop the inbound rate estimate is dropped while the command
   points off the stop, so a one-count phantom rate cannot command torque into
   the stop.
5. The inner state starts with `last_inner_s` set to `None`.

## Errors and faults

`reference_rate` returns `Err(COMMAND_INVALID)` on a failed
`validate_reference` or a nonfinite encoder elevation; `inner_step` degrades
that to a zero tick so no torque leaves the core. The shell latches
containment and publishes the fault.

## Messages

None. The core returns records; the app shell publishes any telemetry.

## Configuration

The core reads the `inner`, `position`, and `integrity` slices of
`ControllerConfig` plus `GimbalConfig` plant and envelope values. The
`science_boundary_guard_deg` offset bounds `RateReference` travel; `K_pos` and
`r_max_deg_per_s` bound the position loop.

## Constraints

The module performs no I/O, bus access, or clock reads. State records are
frozen dataclasses. Time, encoder samples, and the control reference are
arguments. The servo is mode-free: graphs emit references and the shell maps
activations; nothing here branches on a mode or graph name.

## Related documents

- [`flight.payload.gimbal`](gimbal.md)
- [`flight.payload.gimbal.request`](gimbal/request.md)
- [`flight.payload.state`](state.md)
- [`flight.payload.app`](app.md)
