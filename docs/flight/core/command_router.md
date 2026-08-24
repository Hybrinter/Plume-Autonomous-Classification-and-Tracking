# flight.core.command_router

**Source:** `packages/flight/src/flight/core/command_router.py`
**Kind:** module

## Purpose

The command router service drains validated `CommandMsg` values from the bus, runs the pure
routing core, and publishes routed commands, acks, and fault events.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SUBSYSTEM` | constant | Subsystem name `"command_router"` |
| `RouterState` | class | Mutable armed-state map and SAFE-latch flag |
| `CommandRouter` | class | Routing service with `from_config`, `tick`, and `run` |

## Inputs and outputs

**`CommandRouter.from_config(cfg, bus, clock) -> CommandRouter`**

- Inputs: `PactConfig`, shared `MessageBus`, `Clock`.
- Output: router with fresh subscriptions and empty state.

**`CommandRouter.tick() -> None`**

- Drains `SafetyStateMsg` and `CommandMsg` subscriptions and publishes routing outcomes.

**`CommandRouter.run(stop_event) -> None`**

- Input: shutdown `threading.Event`.
- Runs the periodic routing loop with heartbeats until stop.

## Behavior

1. `from_config` subscribes to `CommandMsg` and `SafetyStateMsg`.
2. `from_config` loads routable targets and hazardous command IDs from the command dictionary.
3. Each `tick` updates `safe_latched` from the latest `SafetyStateMsg`.
4. Each `tick` calls `route_command` for every pending `CommandMsg`.
5. The shell stamps wall-clock timestamps on returned messages and publishes them.
6. When `result.routed_command` is set, publish `RoutedCommandMsg`.
7. When `result.ack` is set, publish `CommandAckMsg`.
8. When `result.unroutable_detail` is set, publish `FaultEventMsg(COMMAND_UNROUTABLE)`.
9. `run` calls `tick` on each loop iteration and emits `HeartbeatMsg` every
   `fault.watchdog_interval_s`.

## Errors and faults

Publishes `FaultEventMsg` with `FaultCode.COMMAND_UNROUTABLE` when the pure core returns an
unroutable target detail.

## Messages

**Subscribes:** `CommandMsg`, `SafetyStateMsg`.

**Publishes:** `RoutedCommandMsg`, `CommandAckMsg`, `FaultEventMsg(COMMAND_UNROUTABLE)`,
`HeartbeatMsg`.

## Configuration

| Field | Source |
| --- | --- |
| `command_router.arm_window_s` | ARM expiry window for hazardous commands |
| `fault.watchdog_interval_s` | Heartbeat interval |

## Constraints

- Routing decisions are pure. The shell owns bus, clock, and mutable `RouterState`.
- Unroutable commands produce a loud fault. They are not silently dropped.
- The router emits heartbeats like other persistent-loop core services.

## Related documents

- [`flight.core`](../core.md)
- [`flight.core.routing`](routing.md)
- [`flight.core.composition`](composition.md)
