# flight.mechanical

**Source:** `packages/flight/src/flight/mechanical`
**Kind:** subsystem app

## Purpose

The mechanical package owns the launch lock. It publishes lock state, executes a release
command with a gimbal motion interlock, and emits periodic heartbeats.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`app`](mechanical/app.md) | app shell | Launch lock read, release actuation, and motion interlock |

## Package interface

None.

## Interactions

The app uses the `LaunchLock` HAL protocol. It subscribes to `RoutedCommandMsg` and
`GimbalCommandMsg`. It publishes `LaunchLockStateMsg`, `TelemetryEventMsg`, `HeartbeatMsg`,
`CommandAckMsg`, and `FaultEventMsg`.

The payload app inhibits gimbal motion while the lock is engaged. This app inhibits lock
release while the gimbal is moving.

## Constraints

`RELEASE_LAUNCH_LOCK` is a hazardous command gated by the core command router. The mechanical
app enforces one direction of the bidirectional interlock at actuation time.

## Related documents

- [`flight.payload`](payload.md)
- [`flight.fault`](fault.md)
