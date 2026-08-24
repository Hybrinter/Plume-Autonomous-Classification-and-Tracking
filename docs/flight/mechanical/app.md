# flight.mechanical.app

**Source:** `packages/flight/src/flight/mechanical/app.py`
**Kind:** app shell

## Purpose

`MechanicalApp` reads launch lock state each cycle and publishes it on the bus. It executes
`RELEASE_LAUNCH_LOCK` when the gimbal is not moving and emits periodic heartbeats.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `MechanicalState` | class | Mutable holder of the most recently published lock state |
| `MechanicalApp` | class | Frozen holder of config, bus, clock, lock, and subscriptions |
| `MechanicalApp.from_config` | function | Builds the app and subscribes to routed commands and gimbal commands |
| `MechanicalApp.tick` | method | Handles release commands and publishes lock state |
| `MechanicalApp.run` | method | Periodic loop with heartbeats until the stop event is set |

## Inputs and outputs

`from_config(cfg, bus, clock, lock)` returns a `MechanicalApp`.

`tick()` publishes messages on the bus and returns nothing.

## Behavior

1. Drain `GimbalCommandMsg` values for this cycle. Set a motion flag when any command
   requests ABSOLUTE, STOW, HOME, or non-zero RATE motion.
2. Drain routed commands targeting `mechanical`.
3. For `RELEASE_LAUNCH_LOCK`, reject when the gimbal is moving. Call `lock.release()` when
   motion is absent. Publish an accepted ack on success.
4. Reject any other command targeting mechanical with `COMMAND_INVALID`.
5. Read lock state via `lock.read_state()`. Publish `LaunchLockStateMsg` and a telemetry
   event with the state value. Use `UNKNOWN` when the read returns `Err`.
6. Publish a heartbeat every `watchdog_interval_s`.

## Errors and faults

The app publishes faults and rejected acks; it does not return `Err` from public methods.

| Fault code | Trigger |
| --- | --- |
| `LAUNCH_LOCK_FAULT` | Release inhibited by gimbal motion or driver release failure |
| `COMMAND_INVALID` | Unsupported command targeting mechanical |

## Messages

| Direction | Type |
| --- | --- |
| Subscribe | `RoutedCommandMsg`, `GimbalCommandMsg` |
| Publish | `LaunchLockStateMsg`, `TelemetryEventMsg`, `HeartbeatMsg`, `CommandAckMsg`, `FaultEventMsg` |

## Configuration

| Field | Source | Use |
| --- | --- | --- |
| `watchdog_interval_s` | `FaultConfig` | Tick interval and heartbeat pacing in `run` |

## Constraints

`MechanicalApp` is frozen. `MechanicalState` holds the last published lock state for
inspection. Gimbal motion is inferred from commands seen in the current tick only.
`LAUNCH_LOCK_FAULT` is not a SAFE-triggering fault code.

## Related documents

- [`flight.mechanical`](../mechanical.md)
- [`flight.payload.app`](../payload/app.md)
