# flight.thermal

**Source:** `packages/flight/src/flight/thermal`
**Kind:** subsystem app

## Purpose

The thermal package samples payload temperature and reports housekeeping telemetry. It emits
a fault when the reading exceeds the configured limit and accepts a ground limit override.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`app`](thermal/app.md) | app shell | Temperature sample loop with heartbeat and command handling |

## Package interface

The package re-exports `ThermalApp`.

## Interactions

The app uses the `ScalarSensor` HAL protocol for temperature in degrees Celsius. It
subscribes to `RoutedCommandMsg`. It publishes `TelemetryEventMsg`, `FaultEventMsg`,
`HeartbeatMsg`, and `CommandAckMsg`.

## Constraints

The thermal app is a minimal housekeeping node. It proves the four-channel pattern: heartbeat,
telemetry, command ack, and threshold fault. Limit fields live in `FaultConfig`, not a
thermal-specific config. A sensor read error skips the cycle with no telemetry and no fault.

## Related documents

- [`flight.electrical`](electrical.md)
- [`flight.fault`](fault.md)
