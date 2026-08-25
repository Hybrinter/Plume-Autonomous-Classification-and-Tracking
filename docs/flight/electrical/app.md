# flight.electrical.app

**Source:** `packages/flight/src/flight/electrical/app.py`
**Kind:** app shell

## Purpose

`ElectricalApp` is the electrical housekeeping app. It reads power draw from a scalar sensor,
publishes telemetry, raises `POWER_OVER_LIMIT` when over the limit, rejects routed commands, and
emits periodic heartbeats.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ElectricalApp` | class | Frozen electrical app with config, bus, clock, sensor, and subscription |
| `ElectricalApp.from_config` | function | Builds the app and subscribes to routed commands |
| `ElectricalApp.sample` | method | Reads power draw, publishes telemetry, checks the limit |
| `ElectricalApp.handle_commands` | method | Rejects routed commands targeting `electrical` |
| `ElectricalApp.run` | method | Periodic loop with heartbeats until `stop_event` is set |

## Inputs and outputs

- `from_config(cfg, bus, clock, sensor)` returns an `ElectricalApp`.
- `sample()` and `handle_commands()` publish to the bus; they return nothing.
- `run(stop_event)` runs until the event is set.

## Behavior

1. Drain routed commands targeting `electrical`.
2. Ack each with `REJECTED` and `COMMAND_INVALID` (no commandable behavior exists).
3. Read the sensor; skip the cycle on read error.
4. Publish an `electrical_sample` telemetry event with `power_w`.
5. When power exceeds `power_limit_w`, publish `FaultEventMsg(POWER_OVER_LIMIT)`.
6. Emit `HeartbeatMsg` every `watchdog_interval_s`.

## Errors and faults

| FaultCode | Trigger |
| --- | --- |
| `POWER_OVER_LIMIT` | Power draw exceeds `power_limit_w` |
| `COMMAND_INVALID` | Any routed command targeting `electrical` |

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
| `power_limit_w` | Over-limit threshold in Watts |
| `watchdog_interval_s` | Loop wait and heartbeat interval |

## Constraints

- Unit meaning (Watts) is owned by this app, not the sensor driver.
- Commands for other targets are drained and dropped without re-queue.
- The loop uses `stop_event.wait(timeout=...)` for immediate shutdown.

## Related documents

- [`flight.electrical`](../electrical.md)
- [`flight.thermal`](../thermal.md)
- [`flight.fault.policy`](../fault/policy.md)
