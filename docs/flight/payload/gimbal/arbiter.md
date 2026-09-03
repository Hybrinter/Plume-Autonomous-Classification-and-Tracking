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
| `ArbiterState` | dataclass | Immutable FSM snapshot: mode, blobs, timers, miss count |
| `GimbalArbiter` | class | Stateless arbiter holding `ControllerConfig` and `GimbalConfig` |
| `GimbalArbiter.step` | method | Advances the FSM one frame |

## Inputs and outputs

`GimbalArbiter(cfg, gimbal)` stores controller thresholds and science-limb elevation.

`step(state, result, error_deg, now, safe_commanded, safe_cleared, el_deg=None)` returns
`(ArbiterState, GimbalRequest | None, list[TelemetryEventMsg])`.

## Behavior

1. Enter SAFE and issue STOW when `safe_commanded` is true or `mode_flags` is nonzero.
2. While in SAFE, produce no commands unless `safe_cleared` returns the machine to IDLE.
3. From IDLE: move to ACQUIRING or TRACKING when blobs appear; otherwise stay IDLE.
4. From ACQUIRING: return to IDLE on miss; enter TRACKING when persistence reaches
   `acquire_persistence_frames`.
5. From TRACKING: count consecutive misses. After `release_persistence_frames`, enter
   REWIND when elevation is below the science limb. At the limb, stay TRACKING with no
   command.
6. From REWIND: issue ABSOLUTE elevation to `el_science_max_deg`. Move to ACQUIRING or
   TRACKING when blobs appear. At the limb with no blob, return to TRACKING.
7. In TRACKING with boresight error, issue RATE commands proportional to error, rate
   limited by `retarget_rate_limit_hz` and clamped by hardware slew.

## Errors and faults

None directly. Encoder-runaway escalation happens in `PayloadController`.

## Messages

Returns `TelemetryEventMsg` with event name `state_transition` and subsystem
`controller`. The app shell publishes them.

## Configuration

Reads `ControllerConfig` persistence frames, retarget rate, and Kalman timestep used for
idle accumulation. Reads `GimbalConfig` science-limb elevation and hardware slew.

## Constraints

`step` is a pure function aside from timestamp strings on returned telemetry events.
`GimbalArbiter` holds no mutable instance state; `ArbiterState` threads externally.
LQR refinement of RATE commands happens in `PayloadController` after the arbiter step.

## Related documents

- [`flight.payload.gimbal.request`](request.md)
- [`flight.payload.gimbal.safety`](safety.md)
- [`flight.payload.control`](../control.md)
