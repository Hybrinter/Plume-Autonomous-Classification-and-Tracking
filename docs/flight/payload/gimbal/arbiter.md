# flight.payload.gimbal.arbiter

**Source:** `packages/flight/src/flight/payload/gimbal/arbiter.py`
**Kind:** pure module

## Purpose

The gimbal arbiter is a pure FSM that decides gimbal mode and emits at most one `GimbalRequest`
per frame.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ArbiterState` | class | Immutable FSM snapshot: state, blobs, timers, scan, misses |
| `GimbalArbiter` | class | Stateless arbiter holding `ControllerConfig` |
| `GimbalArbiter.step` | function | Advances the FSM one frame |

## Inputs and outputs

`GimbalArbiter(cfg)` stores config.

`step(state, result, error_deg, now, safe_commanded, safe_cleared)` returns
`(ArbiterState, GimbalRequest | None, list[TelemetryEventMsg])`.

## Behavior

1. On `safe_commanded` or non-zero `result.mode_flags`, latch SAFE, emit STOW, and log a
   transition.
2. While SAFE, produce no requests unless `safe_cleared` returns the machine to IDLE.
3. From IDLE: blobs move to ACQUIRING or TRACKING; empty frames accumulate idle time and may
   enter SCAN after `scan_entry_idle_seconds`.
4. From ACQUIRING: loss of blobs returns to IDLE; persistence threshold enters TRACKING.
5. From TRACKING: blobs reset miss count; consecutive misses at `release_persistence_frames`
   return to IDLE.
6. From SCAN: blobs move to ACQUIRING or TRACKING.
7. In TRACKING with blobs and error degrees, issue RATE at gain 1.0 on error, rate-limited.
8. In SCAN, issue ABSOLUTE azimuth steps that reverse at ±30 deg.

State transitions publish `TelemetryEventMsg` with event name `state_transition`.

## Errors and faults

None. The arbiter returns requests and telemetry only.

## Messages

Returns `TelemetryEventMsg` values for the app shell to publish. Does not publish directly.

## Configuration

Reads `ControllerConfig`: persistence frames, scan timing, slew limits, retarget rate limit.

## Constraints

Pure function: no I/O, no clock reads inside the class. `now` is monotonic seconds used only
for deltas. `_SCAN_LIMIT_DEG` is 30.0.

## Related documents

- [`flight.payload.gimbal`](gimbal.md)
- [`flight.payload.gimbal.request`](request.md)
- [`flight.payload.gimbal.safety`](safety.md)
