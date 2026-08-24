# flight.fault.app

**Source:** `packages/flight/src/flight/fault/app.py`
**Kind:** app shell

## Purpose

`FaultApp` runs the FDIR loop. Each tick it drains heartbeats, routes fault events, checks
the watchdog, handles EXIT_SAFE commands, and publishes the current safety state.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SafetyLatch` | class | Mutable SAFE latch and the fault code that latched it |
| `FaultApp` | class | Frozen holder of config, bus, clock, and subscriptions |
| `FaultApp.from_config` | function | Builds the app and subscribes to heartbeats, faults, and routed commands |
| `FaultApp.initial_entries` | method | Seeds watchdog entries for all monitored subsystems |
| `FaultApp.tick` | method | Runs one watchdog and fault-routing cycle |
| `FaultApp.run` | method | Periodic loop until the stop event is set |

## Inputs and outputs

`from_config(cfg, bus, clock, monitored)` returns a `FaultApp`. `initial_entries()` returns
a watchdog entries dict keyed by subsystem name. `tick(entries, now)` takes the current
entries and monotonic seconds; it returns the updated entries dict.

## Behavior

1. Drain all pending `HeartbeatMsg` values. Reset miss count and update last heartbeat time
   for each monitored subsystem.
2. Drain all pending `FaultEventMsg` values. Call `decide_mode_change` for each event. Publish
   a `ModeChangeMsg(SAFE)` and latch SAFE when the fault code is SAFE-triggering.
3. Call `check_heartbeats` with the current entries. Route each `WATCHDOG_EXPIRE` fault the
   same way as an external fault.
4. Drain routed `EXIT_SAFE` commands. Un-latch SAFE and publish `ModeChangeMsg(IDLE)` when
   SAFE is latched and no SAFE-triggering fault fired this tick. Publish a rejected ack when
   the exit is refused.
5. Publish `SafetyStateMsg` with the current latch state and active faults for this tick.

## Errors and faults

None. The app shell publishes faults and mode changes; it does not return `Err`.

## Messages

| Direction | Type |
| --- | --- |
| Subscribe | `HeartbeatMsg`, `FaultEventMsg`, `RoutedCommandMsg` |
| Publish | `ModeChangeMsg`, `SafetyStateMsg`, `CommandAckMsg` |

## Configuration

| Field | Source | Use |
| --- | --- | --- |
| `watchdog_interval_s` | `FaultConfig` | Tick interval and heartbeat pacing in `run` |
| `watchdog_max_miss_count` | `FaultConfig` | Consecutive misses before `WATCHDOG_EXPIRE` |

The monitored subsystem list is injected at construction; it is not read from config.

## Constraints

`FaultApp` is frozen. The bus, clock, subscriptions, and `SafetyLatch` are mutable services
injected by the composition root. `tick` takes `now` explicitly for deterministic tests.
Monotonic time drives interval math; wall-clock ISO strings stamp published messages only.

## Related documents

- [`flight.fault`](../fault.md)
- [`flight.fault.policy`](policy.md)
- [`flight.fault.watchdog`](watchdog.md)
