# flight.thermal.app

**Source:** `packages/flight/src/flight/thermal/app.py`
**Kind:** app shell

## Purpose

`ThermalApp` reads temperature each cycle, publishes a telemetry event, and raises
`THERMAL_OVER_LIMIT` when the reading exceeds the active limit. It executes routed commands
and emits periodic heartbeats.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ThermalState` | class | Mutable limit override from ground commands |
| `ThermalApp` | class | Frozen holder of config, bus, clock, sensor, and command subscription |
| `ThermalApp.from_config` | function | Builds the app and subscribes to routed commands |
| `ThermalApp.sample` | method | Reads the sensor and publishes telemetry or a fault |
| `ThermalApp.handle_commands` | method | Executes routed commands targeting thermal |
| `ThermalApp.run` | method | Periodic loop with heartbeats until the stop event is set |

## Inputs and outputs

`from_config(cfg, bus, clock, sensor)` returns a `ThermalApp`.

`sample()` and `handle_commands()` publish messages on the bus and return nothing.

## Behavior

1. Drain routed commands targeting `thermal`.
2. For `SET_THERMAL_LIMIT`, store the new limit in `ThermalState` and publish an accepted ack.
3. Reject any other command targeting thermal with `COMMAND_INVALID`.
4. Read the temperature sensor. Skip the cycle when the read returns `Err`.
5. Publish a `TelemetryEventMsg` with `event_name=thermal_sample` and `temperature_c`.
6. When `temperature_c` exceeds the active limit, publish `FaultEventMsg(THERMAL_OVER_LIMIT)`.
7. Publish a heartbeat every `watchdog_interval_s`.

The active limit is `ThermalState.limit_c_override` when set, else `cfg.thermal_limit_c`.

## Errors and faults

The app publishes `FaultEventMsg(THERMAL_OVER_LIMIT)` when temperature exceeds the limit.
It does not return `Err` from public methods.

| Fault code | Trigger |
| --- | --- |
| `THERMAL_OVER_LIMIT` | Temperature above the active limit |
| `COMMAND_INVALID` | Unsupported command targeting thermal |

## Messages

| Direction | Type |
| --- | --- |
| Subscribe | `RoutedCommandMsg` |
| Publish | `TelemetryEventMsg`, `FaultEventMsg`, `HeartbeatMsg`, `CommandAckMsg` |

## Configuration

| Field | Source | Use |
| --- | --- | --- |
| `thermal_limit_c` | `FaultConfig` | Default over-limit threshold in degrees Celsius |
| `watchdog_interval_s` | `FaultConfig` | Sample loop interval and heartbeat pacing |

## Constraints

`ThermalApp` is frozen. `ThermalState` holds the mutable limit override. The sensor returns
a bare float; the app interprets it as degrees Celsius. There is no dedicated sensor-fault
code on read failure.

## Related documents

- [`flight.thermal`](../thermal.md)
- [`flight.electrical.app`](../electrical/app.md)
