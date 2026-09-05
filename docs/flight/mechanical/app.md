# flight.mechanical.app

**Source:** `packages/flight/src/flight/mechanical/app.py`
**Kind:** app shell

## Purpose

`MechanicalApp` owns the launch lock HAL device. It publishes lock state each tick, executes
`RELEASE_LAUNCH_LOCK` when routed, and refuses release while gimbal motion is active.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `MechanicalState` | class | Mutable state holding the last published lock state |
| `MechanicalApp` | class | Frozen mechanical app with lock, bus, clock, and subscriptions |
| `MechanicalApp.from_config` | function | Builds the app and subscribes to routed commands, pose commands, and pointing |
| `MechanicalApp.tick` | method | Handles release commands and publishes lock state |
| `MechanicalApp.run` | method | Periodic loop with heartbeats until `stop_event` is set |

## Inputs and outputs

- `from_config(cfg, bus, clock, lock)` returns a `MechanicalApp`.
- `tick()` publishes lock state and command acks; it returns nothing.
- `run(stop_event)` runs until the event is set.

## Behavior

1. Drain `GimbalCommandMsg` pose commands (`ABSOLUTE`, `STOW`, `HOME`) and payload
   `pointing` telemetry with `|r| > 1e-6`. Either marks the gimbal as moving.
   A locked payload publishes `r=0`.
2. Drain routed commands targeting `mechanical`.
3. On `RELEASE_LAUNCH_LOCK`, reject when gimbal motion is active; otherwise call `lock.release()`.
4. Reject unsupported commands with `COMMAND_INVALID`.
5. Read lock state from the HAL driver and publish `LaunchLockStateMsg` plus a telemetry event.
6. Emit `HeartbeatMsg` every `watchdog_interval_s`.

## Errors and faults

| FaultCode | Trigger |
| --- | --- |
| `LAUNCH_LOCK_FAULT` | Release inhibited by gimbal motion, or HAL release failure |
| `COMMAND_INVALID` | Unsupported command targeting `mechanical` |

## Messages

| Direction | Type |
| --- | --- |
| Subscribe | `RoutedCommandMsg`, `GimbalCommandMsg`, `TelemetryEventMsg` |
| Publish | `LaunchLockStateMsg`, `TelemetryEventMsg`, `FaultEventMsg`, `CommandAckMsg`, `HeartbeatMsg` |

## Configuration

Reads `FaultConfig` via `cfg.fault`:

| Field | Use |
| --- | --- |
| `watchdog_interval_s` | Loop wait and heartbeat interval |

## Constraints

- Motion detection drains the gimbal-command subscription each tick; stale commands do not carry
  over.
- Lock state reads that fail publish `LaunchLockState.UNKNOWN`.
- `RELEASE_LAUNCH_LOCK` is a hazardous command gated by the core command router before routing.
- The loop uses `stop_event.wait(timeout=...)` for immediate shutdown.

## Related documents

- [`flight.mechanical`](../mechanical.md)
- [`flight.payload.gimbal`](../payload/gimbal.md)
- [`flight.hal.interfaces.launch_lock`](../hal/interfaces/launch_lock.md)
