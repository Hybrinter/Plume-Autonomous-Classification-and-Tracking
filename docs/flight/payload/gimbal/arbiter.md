# flight.payload.gimbal.arbiter

**Source:** `packages/flight/src/flight/payload/gimbal/arbiter.py`
**Kind:** pure module

## Purpose

`GimbalArbiter` is the gimbal pointing FSM. It decides gimbal mode and emits at most
one `GimbalRequest` per outer tick. State transitions produce `TelemetryEventMsg`
records for the caller to publish.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ArbiterState` | dataclass | Immutable FSM snapshot: mode, blobs, timers, scan state |
| `GimbalArbiter` | class | Stateless arbiter holding `ControllerConfig` |
| `GimbalArbiter.step` | method | Advances the FSM one outer tick |

## Inputs and outputs

`GimbalArbiter(cfg)` stores controller thresholds.

`step(state, result, error_deg, now, safe_commanded, safe_cleared, dt=None,
has_new_frame=True)` returns `(ArbiterState, GimbalRequest | None,
list[TelemetryEventMsg])`.

## Behavior

1. Enter SAFE and issue STOW when `safe_commanded` is true or `mode_flags` is nonzero.
2. While in SAFE, produce no commands unless `safe_cleared` returns the machine to IDLE.
3. From IDLE: move to ACQUIRING or TRACKING when a new frame has blobs; add `dt` to idle
   time and enter SCAN when idle exceeds `scan_entry_idle_seconds`.
4. From ACQUIRING: on a new frame, return to IDLE on miss; enter TRACKING when
   persistence reaches `acquire_persistence_frames`.
5. From TRACKING: on a new frame, count consecutive misses; release to IDLE after
   `release_persistence_frames`. Outer ticks with no new frame do not count as misses.
6. From SCAN: on a new frame with blobs, move to ACQUIRING or TRACKING; issue ABSOLUTE
   pan steps on a raster that reverses at +/-30 degrees azimuth. Pan increment uses
   `dt` (or `1 / retarget_rate_limit_hz` when `dt` is omitted).
7. In TRACKING with boresight error, issue RATE commands proportional to error on every
   tick, clamped by `max_slew_rate_deg_per_s`.

## Errors and faults

None directly.

## Messages

Returns `TelemetryEventMsg` with event name `state_transition` and subsystem
`controller`. The app shell publishes them.

## Configuration

Reads `ControllerConfig`: persistence frames, scan timing and slew, retarget rate
(SCAN default `dt` only), and max slew rate.

## Constraints

`step` is a pure function aside from timestamp strings on returned telemetry events.
`GimbalArbiter` holds no mutable instance state; `ArbiterState` threads externally.
LQR overwrite of RATE commands happens in `PayloadController.step_outer` after the
arbiter step.

## Related documents

- [`flight.payload.gimbal.request`](request.md)
- [`flight.payload.gimbal.safety`](safety.md)
- [`flight.payload.control`](../control.md)
