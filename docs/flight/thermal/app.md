# flight.thermal.app

**Source:** `packages/flight/src/flight/thermal/app.py`
**Kind:** app shell

## Purpose

`ThermalApp` is the thermal housekeeping app. It reads temperature from a scalar sensor, publishes
telemetry, raises `THERMAL_OVER_LIMIT` when over the limit, executes `SET_THERMAL_LIMIT`, and
emits periodic heartbeats.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ThermalState` | class | Mutable state holding an optional limit override |
| `ThermalApp` | class | Frozen thermal app with config, bus, clock, sensor, and subscription |
| `ThermalApp.from_config` | function | Builds the app and subscribes to routed commands |
| `ThermalApp.sample` | method | Reads temperature, publishes telemetry, checks the limit |
| `ThermalApp.handle_commands` | method | Executes routed commands targeting `thermal` |
| `ThermalApp.run` | method | Periodic loop with heartbeats until `stop_event` is set |

## Inputs and outputs

- `from_config(cfg, bus, clock, sensor)` returns a `ThermalApp`.
- `sample()` and `handle_commands()` publish to the bus; they return nothing.
- `run(stop_event)` runs until the event is set.

## Behavior

1. Drain routed commands targeting `thermal`.
2. On `SET_THERMAL_LIMIT`, store the new threshold in `ThermalState.limit_c_override` and ack
   `ACCEPTED`.
3. Reject any other command targeting `thermal` with `COMMAND_INVALID`.
4. Read the sensor; skip the cycle on read error.
5. Publish a `thermal_sample` telemetry event with `temperature_c`.
6. When temperature exceeds the active limit, publish `FaultEventMsg(THERMAL_OVER_LIMIT)`.
7. Emit `HeartbeatMsg` every `watchdog_interval_s`.

## Errors and faults

| FaultCode | Trigger |
| --- | --- |
| `THERMAL_OVER_LIMIT` | Temperature exceeds the active limit |
| `COMMAND_INVALID` | Unsupported command targeting `thermal` |

Sensor read errors produce no fault code; missing telemetry is observable by the watchdog and
ground.

## Messages

| Direction | Type |
| --- | --- |
| Subscribe | `RoutedCommandMsg` |
| Publish | `TelemetryEventMsg`, `FaultEventMsg`, `CommandAckMsg`, `HeartbeatMsg` |

## Configuration

Reads `FaultConfig` via `cfg.fault`:

| Field | Use |
| --- | --- |
| `thermal_limit_c` | Default over-limit threshold in degrees Celsius |
| `watchdog_interval_s` | Loop wait and heartbeat interval |

## Constraints

- The active limit is `ThermalState.limit_c_override` when set, else `thermal_limit_c`.
- Unit meaning (Celsius) is owned by this app, not the sensor driver.
- The loop uses `stop_event.wait(timeout=...)` for immediate shutdown.

## Related documents

- [`flight.thermal`](../thermal.md)
- [`flight.electrical`](../electrical.md)
- [`flight.fault.policy`](../fault/policy.md)
