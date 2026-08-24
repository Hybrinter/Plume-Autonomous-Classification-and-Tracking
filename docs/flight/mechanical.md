# flight.mechanical

**Source:** `packages/flight/src/flight/mechanical`
**Kind:** subsystem app

## Purpose

The mechanical package owns the launch lock device. Each cycle it publishes lock state, executes
`RELEASE_LAUNCH_LOCK` with a gimbal-motion interlock, and sends heartbeats.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`app`](mechanical/app.md) | module | Launch-lock owner and release actuation |

## Package interface

None. The package has no `__init__.py` re-exports; the composition root imports `MechanicalApp`
from `flight.mechanical.app`.

## Interactions

Uses the `LaunchLock` HAL protocol. Subscribes to `RoutedCommandMsg` and `GimbalCommandMsg`.
Publishes `LaunchLockStateMsg`, `TelemetryEventMsg`, `FaultEventMsg`, `CommandAckMsg`, and
`HeartbeatMsg`. The payload app consumes `LaunchLockStateMsg` and inhibits gimbal motion while the
lock is engaged.

## Constraints

- Release is inhibited when gimbal motion is commanded in the same tick.
- `LAUNCH_LOCK_FAULT` is not in the SAFE-triggering fault set.
- The bidirectional interlock spans mechanical (release side) and payload (motion side).
- The fault app monitors this subsystem via heartbeats when listed in `MONITORED_SUBSYSTEMS`.

## Related documents

- [`flight.payload.gimbal`](payload/gimbal.md)
- [`flight.fault`](fault.md)
- [`flight.hal.interfaces.launch_lock`](hal/interfaces/launch_lock.md)
