# flight.electrical

**Source:** `packages/flight/src/flight/electrical`
**Kind:** subsystem app

## Purpose

The electrical package samples payload power draw and reports housekeeping telemetry. It
emits a fault when the reading exceeds the configured limit.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`app`](electrical/app.md) | app shell | Power sample loop with heartbeat and command rejection |

## Package interface

The package re-exports `ElectricalApp`.

## Interactions

The app uses the `ScalarSensor` HAL protocol for power in Watts. It subscribes to
`RoutedCommandMsg`. It publishes `TelemetryEventMsg`, `FaultEventMsg`, `HeartbeatMsg`, and
`CommandAckMsg`.

## Constraints

The electrical app matches the thermal app structure. It has no commandable behavior in the
command dictionary. Limit fields live in `FaultConfig`. A sensor read error skips the cycle
with no telemetry and no fault.

## Related documents

- [`flight.thermal`](thermal.md)
- [`flight.fault`](fault.md)
