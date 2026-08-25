# flight.electrical

**Source:** `packages/flight/src/flight/electrical`
**Kind:** subsystem app

## Purpose

The electrical package runs housekeeping for the electrical node. Each cycle it samples power draw,
publishes telemetry, emits an over-limit fault when the reading exceeds the threshold, rejects any
routed commands, and sends heartbeats.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`app`](electrical/app.md) | module | Electrical housekeeping app shell |

## Package interface

Re-exports `ElectricalApp`.

## Interactions

Uses the `ScalarSensor` HAL protocol for power draw in Watts. Subscribes to `RoutedCommandMsg`.
Publishes `TelemetryEventMsg`, `FaultEventMsg`, `CommandAckMsg`, and `HeartbeatMsg`. The fault app
monitors this subsystem via heartbeats.

## Constraints

- The power limit comes from `FaultConfig.power_limit_w`.
- A sensor read error skips the cycle with no telemetry and no fault.
- `POWER_OVER_LIMIT` is in the SAFE-triggering fault set.
- No commandable behavior exists in the command dictionary for this node.
- The app does not cross-import other peer subsystem packages.

## Related documents

- [`flight.thermal`](thermal.md)
- [`flight.fault`](fault.md)
- [`flight.hal.interfaces.scalar`](hal/interfaces/scalar.md)
