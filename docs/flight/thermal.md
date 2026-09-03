# flight.thermal

**Source:** `packages/flight/src/flight/thermal`
**Kind:** subsystem app

## Purpose

The thermal package runs housekeeping for the thermal node. Each cycle it samples temperature,
publishes telemetry, handles routed commands, and sends heartbeats. Datasheet limits live on
`ThermalConfig` as records. The app does not emit `THERMAL_OVER_LIMIT`.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`app`](thermal/app.md) | module | Thermal housekeeping app shell |

## Package interface

Re-exports `ThermalApp`.

## Interactions

Uses the `ScalarSensor` HAL protocol for temperature in degrees Celsius. Subscribes to
`RoutedCommandMsg`. Publishes `TelemetryEventMsg`, `CommandAckMsg`, and `HeartbeatMsg`.
The fault app monitors this subsystem via heartbeats.

## Constraints

- Per-component min/max values live on `ThermalConfig`. `sample()` does not compare them.
- `SET_THERMAL_LIMIT` stores an override for later sensors and does not enable a compare.
- A sensor read error skips the cycle with no telemetry and no fault.
- `THERMAL_OVER_LIMIT` remains in the SAFE-triggering fault set for later sensors.
- The app does not cross-import other peer subsystem packages.

## Related documents

- [`flight.electrical`](electrical.md)
- [`flight.fault`](fault.md)
- [`flight.hal.interfaces.scalar`](hal/interfaces/scalar.md)
