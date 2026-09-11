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
| `ControlState` | dataclass | Arbiter, residual history, inner state, and command state |
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
`OuterTick.state.residual` contains the latest replayed estimate.

## Behavior

1. `ingest_inference` applies confidence and area gates, matches blobs, and
   forms the area-weighted centroid of every accepted component. It stores the
   frame ID and shutter time in the queued sample.
2. `inner_step` keeps the timestamped encoder ring, fits `y_m`, and runs the
   detailed-plant PI. `y_m` remains available for inner integrity checks and
   simulation. It is not an outer residual-estimator input.
3. `outer_step` submits the encoder sample and a nominal-rate sample. It submits
   an explicit reference change when one is present. It submits a vision event
   at its shutter time and requests replay at the current encoder sample time.
4. Residual replay uses encoder angle displacement, encoder uncertainty, and
   reversal uncertainty. A vision event is accepted only when its shutter time
   has an exact encoder sample or a valid bracket.
5. The predictor supplies nominal elevation rate and unactuated azimuth rate of a
   frozen ECEF CoG locked at `cog_height_m`. A smooth sampled change uses the
   zero-order-hold rate history. An explicit reference replacement rebases the
   residual rate and keeps total target rate continuous.
6. The tracking rate law matches `omega_t_nom + omega_t_res` and smear-caps only
   `Kp * e`. REWIND matches boresight-ground `omega_el` and hunts at the
   elevation smear cap for `rewind_sharp_max_s`. After that window it escapes at
   the hardware slew. Residual is ignored in REWIND. Visual tracking can run
   without navigation.
7. STOW, HOME, and ABSOLUTE requests override tracking through the position loop.
   SAFE entry and SAFE exit reset the residual checkpoint. REWIND ignores residual
   rate in the tracking law and does not drop encoder history.
8. The state starts with `last_inner_s` and `last_outer_s` set to `None`.

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
- [`flight.payload.tracking`](tracking.md)
- [`flight.payload.app`](app.md)
