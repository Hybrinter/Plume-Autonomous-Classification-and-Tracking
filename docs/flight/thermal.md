# flight.thermal

**Source:** `packages/flight/src/flight/thermal`
**Kind:** subsystem app

## Purpose

The thermal package runs housekeeping for the thermal node. Each cycle it samples temperature,
publishes telemetry, emits an over-limit fault when the reading exceeds the threshold, handles
routed commands, and sends heartbeats.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`app`](thermal/app.md) | module | Thermal housekeeping app shell |

## Package interface

Re-exports `ThermalApp`.

## Interactions

Uses the `ScalarSensor` HAL protocol for temperature in degrees Celsius. Subscribes to
`RoutedCommandMsg`. Publishes `TelemetryEventMsg`, `FaultEventMsg`, `CommandAckMsg`, and
`HeartbeatMsg`. The fault app monitors this subsystem via heartbeats.

## Constraints

- Threshold limits come from `FaultConfig.thermal_limit_c` unless overridden by `SET_THERMAL_LIMIT`.
- A sensor read error skips the cycle with no telemetry and no fault.
- `THERMAL_OVER_LIMIT` is in the SAFE-triggering fault set.
- The app does not cross-import other peer subsystem packages.

## Related documents

- [`flight.electrical`](electrical.md)
- [`flight.fault`](fault.md)
- [`flight.hal.interfaces.scalar`](hal/interfaces/scalar.md)
