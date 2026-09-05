# flight.payload.gimbal.arbiter

**Source:** `packages/flight/src/flight/payload/gimbal/arbiter.py`
**Kind:** pure module

## Purpose

`GimbalArbiter` selects TRACKING, REWIND, or SAFE. It does not emit axis rates or
torque. SAFE latches until ground clears it.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ArbiterState` | dataclass | Immutable FSM snapshot: mode, blobs, target id, miss count |
| `GimbalArbiter` | class | Stateless arbiter holding `ControllerConfig` and `GimbalConfig` |
| `GimbalArbiter.step` | method | Advances the FSM one outer tick |

## Inputs and outputs

`GimbalArbiter(cfg, gimbal)` stores controller thresholds and science-limb elevation.

`step(state, blobs, now, safe_commanded, safe_cleared, el_deg, mode_flags=0,
vision_updated=True)` returns
`(ArbiterState, GimbalRequest | None, list[TelemetryEventMsg])`.

The request is STOW on SAFE entry. Otherwise it is `None`. The outer law owns `r`.

## Behavior

1. Enter SAFE and issue STOW when `safe_commanded` is true or `mode_flags` is nonzero.
2. While in SAFE, produce no commands unless `safe_cleared` returns the machine to
   TRACKING.
3. From TRACKING: a plume resets miss count. After `release_persistence_frames` empty
   vision samples, enter REWIND when elevation is below the science limb. At the limb,
   stay TRACKING.
4. From REWIND: a plume returns to TRACKING. Arrival at the limb with no plume also
   returns to TRACKING.
5. Outer coast ticks pass `vision_updated=False` and leave `miss_count` unchanged.

## Errors and faults

None directly.

## Messages

Returns `TelemetryEventMsg` with event name `state_transition` and subsystem
`controller`. The app shell publishes them.

## Configuration

Reads `ControllerConfig.release_persistence_frames` and `limb_arrival_deg`. Reads
`GimbalConfig` science-limb and stow elevation.

## Constraints

`step` is a pure function aside from timestamp strings on returned telemetry events.
`GimbalArbiter` holds no mutable instance state; `ArbiterState` threads externally.

## Related documents

- [`flight.payload.gimbal.request`](request.md)
- [`flight.payload.gimbal.outer`](outer.md)
- [`flight.payload.control`](../control.md)
