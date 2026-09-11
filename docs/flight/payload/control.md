# flight.payload.control

**Source:** `packages/flight/src/flight/payload/control.py`

**Kind:** pure module

## Purpose

`PayloadController` is the pure cascaded elevation controller. It exposes
`inner_step` and `outer_step`. The inner path keeps the detailed-plant encoder
rate fit and PI state. The outer path uses timestamped encoder increments,
predictor events, vision replay, and the rate law.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `VisionSample` | dataclass | Frame ID, shutter time, error, centroid, exposure, blobs, and ISS |
| `IssSample` | dataclass | ISS ECI state for the predictor |
| `EncoderState` | dataclass | Timestamped encoder samples, last angle, and measured rate |
| `InnerControlState` | dataclass | Inner PI integrator, last inner time, and last torque |
| `IntegrityState` | dataclass | Freeze and lock-fight strikes with lock-hold latch |
| `TargetState` | dataclass | Stored CoG and last scene-rate terms |
| `PoseState` | dataclass | Position-loop mode and target elevation |
| `ControlState` | dataclass | Nested records grouped by the loop that updates them |
| `InnerTick` | dataclass | Updated state and detailed-plant torque |
| `OuterTick` | dataclass | Updated state, optional pose request, telemetry, and fault |
| `PayloadController` | dataclass | Immutable cascaded control core |
| `PayloadController.from_config` | static method | Builds the control core from typed config |
| `PayloadController.initial_state` | method | Cold TRACKING state and empty residual history |
| `PayloadController.ingest_inference` | method | Gates blobs and creates a vision sample |
| `PayloadController.inner_step` | method | Updates the detailed-plant PI path |
| `PayloadController.outer_step` | method | Submits events and computes the outer rate |

## Inputs and outputs

`from_config` takes controller, sensor, gimbal, ephemeris, and preprocessing
slices. `inner_step` takes a raw encoder angle and optional encoder sample time.
`outer_step` takes an `EncoderSample`, optional `VisionSample`, optional
`IssSample`, SAFE flags, and an optional explicit
`PredictorReferenceChange`.

`OuterTick.state.residual_history` contains the bounded event history.
`OuterTick.state.residual` is the snapshot of `estimate_at` at the last
TRACKING tick. `inner_step` writes `EncoderState`, `InnerControlState`, and
`commanded_rate_rad_s`. `outer_step` writes arbiter, residual, `TargetState`,
`PoseState`, `last_outer_s`, `commanded_rate_rad_s`, and `last_rate_decision`.

## Behavior

1. `ingest_inference` applies confidence and area gates, matches blobs, and
   forms the area-weighted centroid of every accepted component. It stores the
   frame ID and shutter time in the queued sample.
2. `inner_step` appends one `EncoderSample` to `EncoderState.samples`, trims
   the ring to `rate_fit_n`, fits `measured_rate_rad_s`, and runs the
   detailed-plant PI. The measured rate remains available for inner integrity
   checks and simulation. It is not an outer residual-estimator input.
3. `outer_step` updates CoG from vision while TRACKING. It cold-starts the
   residual on TRACKING acquire. An identity reset drops the stored CoG unless
   this frame wrote a new intersect. It then calls `select_scene`. Residual
   encoder, nominal, and vision events run only in TRACKING. REWIND freezes the
   residual and sets `TargetState.r_cog_ecef_m` to `None`.
4. Residual replay uses encoder angle displacement, encoder uncertainty, and
   reversal uncertainty. A vision event is accepted only when its shutter time
   has an exact encoder sample or a valid bracket. TRACKING acquire seeds the
   checkpoint at shutter when the sample carries a shutter encoder angle. A
   missing shutter angle leaves the checkpoint angle unset.
5. `select_scene` supplies nominal elevation rate of a frozen ECEF CoG in
   TRACKING, or of the boresight height-proxy hit in REWIND. Missing ISS is
   unknown navigation. It is not a zero-rate scene. An IoU-matched CoG
   replacement rebases residual rate with the old CoG at the current ISS time.
   ISS motion between ticks is not a reference jump.
6. The tracking/rewind path calls `outer_rate` and stores
   `RateDecision.commanded_rate_rad_s` on `ControlState.commanded_rate_rad_s`.
   It also stores the `RateDecision` on `last_rate_decision`. The pose path
   leaves `last_rate_decision` empty. TRACKING matches
   `omega_t_nom + omega_t_res` and smear-caps only `Kp * e`. REWIND matches
   boresight-ground `omega_el` and hunts at `+omega_sharp` for
   `rewind_sharp_max_s`. After that window it escapes at `+omega_hw`. Residual
   is ignored and is not fed boresight rates. Visual tracking can run without
   navigation. The pose path writes a float `r` from `position_rate`.
7. STOW, HOME, and ABSOLUTE requests override tracking through the position loop.
   SAFE zeros tracking and SAFE exit resets the residual checkpoint. REWIND does
   not drop the inner encoder samples. A single TRACKING miss keeps the residual
   and CoG. Acquire from cold, from REWIND, or from unmatched blob IDs resets the
   residual. That reset also drops the prior CoG unless this frame produced a new
   intersect.
8. The state starts with `inner.last_inner_s` and `last_outer_s` set to `None`.

## Errors and faults

`OuterTick.fault` is `None`; the app shell publishes integrity and HAL faults.
Residual event dispositions are retained by the history API and are available
to telemetry integration.

## Messages

None. The pure core returns `GimbalRequest` and `TelemetryEventMsg` values. The
app shell publishes them.

## Configuration

The controller reads nested vision, arbiter, inner, outer, residual, position,
integrity, and predictor config. `predictor.cog_height_m` is the tracking-proxy
intersect height. `outer.rewind_sharp_max_s` is the sharp REWIND window. The
residual config supplies continuous acceleration density, encoder and reversal
uncertainty, interpolation and timing limits, and history horizon. Gimbal
geometry, plant values, WGS-84 values, and smear budget remain injected typed
config.

## Constraints

The module performs no I/O, bus access, or clock reads. State is immutable. The
inner path remains available for the detailed simulation plant. Production
hardware uses the rate-command HAL path selected by the app shell.

## Related documents

- [`flight.payload.gimbal`](gimbal.md)
- [`flight.payload.gimbal.scene`](gimbal/scene.md)
- [`flight.payload.tracking`](tracking.md)
- [`flight.payload.app`](app.md)
