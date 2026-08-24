# flight.payload.gimbal.arbiter

**Source:** `packages/flight/src/flight/payload/gimbal/arbiter.py`
**Kind:** pure module

## Purpose

`GimbalArbiter` is the gimbal pointing FSM. It decides gimbal mode and emits at most
one `GimbalRequest` per frame. State transitions produce `TelemetryEventMsg` records
for the caller to publish.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ArbiterState` | dataclass | Immutable FSM snapshot: mode, blobs, timers, scan state |
| `GimbalArbiter` | class | Stateless arbiter holding `ControllerConfig` |
| `GimbalArbiter.step` | method | Advances the FSM one frame |

## Inputs and outputs

`GimbalArbiter(cfg)` stores controller thresholds.

`step(state, result, error_deg, now, safe_commanded, safe_cleared)` returns
`(ArbiterState, GimbalRequest | None, list[TelemetryEventMsg])`.

## Behavior

1. Enter SAFE and issue STOW when `safe_commanded` is true or `mode_flags` is nonzero.
2. While in SAFE, produce no commands unless `safe_cleared` returns the machine to IDLE.
3. From IDLE: move to ACQUIRING or TRACKING when blobs appear; accumulate idle time and
   enter SCAN when idle exceeds `scan_entry_idle_seconds`.
4. From ACQUIRING: return to IDLE on miss; enter TRACKING when persistence reaches
   `acquire_persistence_frames`.
5. From TRACKING: count consecutive misses; release to IDLE after
   `release_persistence_frames`.
6. From SCAN: move to ACQUIRING or TRACKING when blobs appear; issue ABSOLUTE pan steps
   on a raster that reverses at +/-30 degrees azimuth.
7. In TRACKING with boresight error, issue RATE commands proportional to error, rate
   limited by `retarget_rate_limit_hz` and clamped by `max_slew_rate_deg_per_s`.

## Errors and faults

None directly. Runaway escalation happens in `PayloadController` via deadband strikes.

## Messages

Returns `TelemetryEventMsg` with event name `state_transition` and subsystem
`controller`. The app shell publishes them.

## Configuration

Reads `ControllerConfig`: persistence frames, scan timing and slew, retarget rate,
max slew rate, and Kalman timestep used for idle accumulation.

## Constraints

`step` is a pure function aside from timestamp strings on returned telemetry events.
`GimbalArbiter` holds no mutable instance state; `ArbiterState` threads externally.
LQR refinement of RATE commands happens in `PayloadController` after the arbiter step.

## Related documents

- [`flight.payload.gimbal.request`](request.md)
- [`flight.payload.gimbal.safety`](safety.md)
- [`flight.payload.control`](../control.md)
