# flight.payload.gimbal.arbiter

**Source:** `packages/flight/src/flight/payload/gimbal/arbiter.py`
**Kind:** pure module

## Purpose

`GimbalArbiter` selects TRACKING, REWIND, or SAFE. It does not emit axis rates or
torque. SAFE latches until ground clears it.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ArbiterState` | dataclass | Immutable FSM snapshot: mode, blobs, aggregate liveness, observation age state, miss count |
| `GimbalArbiter` | class | Stateless arbiter holding `ArbiterConfig` and `GimbalConfig` |
| `GimbalArbiter.step` | method | Advances the FSM one outer tick |

## Inputs and outputs

`GimbalArbiter(cfg, gimbal)` stores arbiter thresholds and science-limb elevation.

`step(state, blobs, now, safe_commanded, safe_cleared, el_deg, mode_flags=0,
vision_updated=True, observation_t_s=None, coast_permitted=True,
timestamp_utc="")` returns
`(ArbiterState, GimbalRequest | None, list[TelemetryEventMsg])`.

The request is STOW on SAFE entry. Otherwise it is `None`. The outer law owns `r`.

## Behavior

1. Enter SAFE and issue STOW when `safe_commanded` is true or `mode_flags` is nonzero.
2. While in SAFE, produce no commands unless `safe_cleared` returns the machine to
   TRACKING.
3. An accepted aggregate enters or returns to TRACKING immediately. Its liveness is
   independent of a component ID.
4. From TRACKING: empty packets increment the release counter. The first of
   `release_persistence_frames` empty packets, `max_observation_age_s` since the
   accepted aggregate, or a false `coast_permitted` expires the coast. Enter REWIND
   below the science limb; at the limb, stay TRACKING with `r = 0`.
5. From REWIND: an accepted aggregate returns to TRACKING. Arrival at the limb with no plume also
   returns to TRACKING.
6. Outer coast ticks pass `vision_updated=False` and leave `miss_count` unchanged,
   but elapsed observation age continues. This prevents a silent vision pipeline
   from preserving a live target indefinitely.

## Errors and faults

None directly.

## Messages

Returns `TelemetryEventMsg` with event name `state_transition` and subsystem
`controller`. The app shell publishes them.

## Configuration

Reads `ArbiterConfig.release_persistence_frames`, `max_observation_age_s`, and
`limb_arrival_deg`. Reads
`GimbalConfig` science-limb and stow elevation.

## Constraints

`step` is a pure function. Transition telemetry uses the injected `timestamp_utc`.
`GimbalArbiter` holds no mutable instance state; `ArbiterState` threads externally.

## Related documents

- [`flight.payload.gimbal.request`](request.md)
- [`flight.payload.gimbal.outer`](outer.md)
- [`flight.payload.control`](../control.md)
