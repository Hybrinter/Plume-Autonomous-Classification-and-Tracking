# flight.payload.control

**Source:** `packages/flight/src/flight/payload/control.py`
**Kind:** pure module

## Purpose

`PayloadController` is the pure cascaded elevation controller. It exposes
`inner_step` and `outer_step`. Inner: encoder ring, polynomial `y_m`, PI plus
computed torque. Outer: CoG intersect, co-rotating predictor, residual filter,
arbiter, and rate reference.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `VisionSample` | dataclass | Queued vision packet: `z_v`, centroid, exposure, blobs, shutter `theta_g`, ISS |
| `IssSample` | dataclass | ISS ECI state for the predictor |
| `ControlState` | dataclass | Bundled arbiter, residual, encoder ring, integrator, CoG, clock origins |
| `InnerTick` | dataclass | Updated state and torque |
| `OuterTick` | dataclass | Updated state, optional STOW request, telemetry |
| `PayloadController` | dataclass | Immutable control core |
| `PayloadController.from_config` | static method | Builds arbiter, residual filter, and pinhole geometry |
| `PayloadController.initial_state` | method | Cold TRACKING, `r = 0`, empty encoder ring |
| `PayloadController.ingest_inference` | method | Gate and match blobs; build a vision sample |
| `PayloadController.inner_step` | method | One inner tick |
| `PayloadController.outer_step` | method | One outer tick |

## Inputs and outputs

`from_config` takes controller, sensor, gimbal, ephemeris, and preprocessing
slices. `inner_step` takes encoder elevation in radians. `outer_step` takes
elevation, optional `VisionSample`, optional `IssSample`, and SAFE flags.

## Behavior

1. `ingest_inference` applies confidence and area gates, matches blobs, and forms
   pinhole `z_v`. It stamps shutter elevation and ISS onto the sample.
2. `inner_step` pushes the encoder sample, fits `y_m`, and runs the inner PI.
   Locked or SAFE/pose ticks freeze or replace `r` before the PI.
3. `outer_step` runs the arbiter, intersects CoG with shutter-stamped pose, predicts
   `omega_t_nom`, predicts/updates the residual filter, and writes `r`. SAFE skips
   rewind and CoG replace. EXIT_SAFE cold-starts the residual.
4. Live TRACKING is `current_target_id is not None` and ISS present, not
   `has_measurement`. Science-window clips zero `r` that would leave
   `[el_science_min, el_science_max]`.
5. STOW / HOME / ABSOLUTE override `r` through the position loop.
6. `last_inner_s` and `last_outer_s` start as `None`.

## Errors and faults

`OuterTick.fault` is always `None`. Integrity trips are published by the app shell.

## Messages

None. The pure core returns `GimbalRequest` and `TelemetryEventMsg` values; the app
shell publishes them.

## Configuration

Reads nested `ControllerConfig` tables (vision, arbiter, inner, outer, residual,
position, integrity), plant copies, encoder counts, WGS-84 scalars, and smear budget.

## Constraints

The module performs no I/O, bus access, or clock reads. Time arrives as `now` and
`dt_s`. State is immutable. Inner units are SI.

## Related documents

- [`flight.payload.gimbal`](gimbal.md)
- [`flight.payload.tracking`](tracking.md)
- [`flight.payload.app`](app.md)
