# flight.electrical.app

**Source:** `packages/flight/src/flight/electrical/app.py`
**Kind:** app shell

## Purpose

`ElectricalApp` reads power draw each cycle, publishes a telemetry event, and raises
`POWER_OVER_LIMIT` when the reading exceeds the configured limit. It rejects any routed
command targeting electrical and emits periodic heartbeats.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ElectricalApp` | class | Frozen holder of config, bus, clock, sensor, and command subscription |
| `ElectricalApp.from_config` | function | Builds the app and subscribes to routed commands |
| `ElectricalApp.sample` | method | Reads the sensor and publishes telemetry or a fault |
| `ElectricalApp.handle_commands` | method | Rejects routed commands targeting electrical |
| `ElectricalApp.run` | method | Periodic loop with heartbeats until the stop event is set |

## Inputs and outputs

`from_config(cfg, bus, clock, sensor)` returns an `ElectricalApp`.

`sample()` and `handle_commands()` publish messages on the bus and return nothing.

## Behavior

1. Drain routed commands targeting `electrical`. Publish a rejected ack for each with
   `COMMAND_INVALID`.
2. Read the power sensor. Skip the cycle when the read returns `Err`.
3. Publish a `TelemetryEventMsg` with `event_name=electrical_sample` and `power_w`.
4. When `power_w` exceeds `cfg.power_limit_w`, publish `FaultEventMsg(POWER_OVER_LIMIT)`.
5. Publish a heartbeat every `watchdog_interval_s`.

## Errors and faults

The app publishes `FaultEventMsg(POWER_OVER_LIMIT)` when power exceeds the limit. It does
not return `Err` from public methods.

| Fault code | Trigger |
| --- | --- |
| `POWER_OVER_LIMIT` | Power above `power_limit_w` |
| `COMMAND_INVALID` | Any command routed to electrical |

## Messages

| Direction | Type |
| --- | --- |
| Subscribe | `RoutedCommandMsg` |
| Publish | `TelemetryEventMsg`, `FaultEventMsg`, `HeartbeatMsg`, `CommandAckMsg` |

## Configuration

| Field | Source | Use |
| --- | --- | --- |
| `power_limit_w` | `FaultConfig` | Over-limit threshold in Watts |
| `watchdog_interval_s` | `FaultConfig` | Sample loop interval and heartbeat pacing |

## Constraints

`ElectricalApp` is frozen. The sensor returns a bare float; the app interprets it as Watts.
There is no dedicated sensor-fault code on read failure. The command router does not route
commands to electrical under normal operation.

## Related documents

- [`flight.electrical`](../electrical.md)
- [`flight.thermal.app`](../thermal/app.md)
