# flight.core.downlink

**Source:** `packages/flight/src/flight/core/downlink.py`
**Kind:** module

## Purpose

The downlink manager is the single prioritized, AOS-gated path from bus events to
`DownlinkItemMsg` for station transmission.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SUBSYSTEM` | constant | Subsystem name `"downlink"` |
| `DownlinkState` | class | Mutable priority queue, order counter, and AOS flag |
| `DownlinkManager` | class | Downlink service with `from_config`, `tick`, and `run` |

## Inputs and outputs

**`DownlinkManager.from_config(cfg, bus, clock) -> DownlinkManager`**

- Inputs: `PactConfig`, shared `MessageBus`, `Clock`.
- Output: manager with subscriptions and an empty LOS queue.

**`DownlinkManager.tick() -> None`**

- Enqueues inbound items and emits within AOS and the per-pass byte budget.

**`DownlinkManager.run(stop_event) -> None`**

- Input: shutdown `threading.Event`.
- Runs the downlink loop with heartbeats until stop.

## Behavior

1. `from_config` subscribes to `FaultEventMsg`, `CommandAckMsg`, `TelemetryEventMsg`,
   `ProductRefMsg`, and `LinkStateMsg`.
2. Each `tick` reads `LinkStateMsg` and sets the AOS flag when state is `LinkState.AOS`.
3. Fault events enqueue as inline JSON at priority `FAULT_EVENT`.
4. Command acks enqueue as inline JSON at priority `COMMAND_ACK`.
5. Telemetry events enqueue as inline JSON at priority `HK_TELEMETRY`.
6. Product refs enqueue with a storage reference, byte length, and the product priority.
7. During loss of signal, items remain in the pending queue. Nothing emits.
8. During AOS, pending items sort by priority then insertion order.
9. Items emit until the per-pass byte budget is reached. The first item always sends even
   when it alone exceeds the budget.
10. Inline items carry payload bytes and CRC-32. Storage-ref items carry an empty payload
    and a storage entry id.
11. `run` calls `tick` each loop and emits `HeartbeatMsg` every `fault.watchdog_interval_s`.

## Errors and faults

None at the library level.

## Messages

**Subscribes:** `FaultEventMsg`, `CommandAckMsg`, `TelemetryEventMsg`, `ProductRefMsg`,
`LinkStateMsg`.

**Publishes:** `DownlinkItemMsg`, `HeartbeatMsg`.

## Configuration

| Field | Source |
| --- | --- |
| `comms.downlink_max_bytes_per_pass` | Per-pass byte budget |
| `fault.watchdog_interval_s` | Heartbeat interval |

## Constraints

- This module is the sole producer of `DownlinkItemMsg`.
- Priority order is fault events, then command acks, then housekeeping telemetry, then
  science products.
- Large science products stay off the bus as storage references.
- The manager emits heartbeats like other persistent-loop core services.

## Related documents

- [`flight.core`](../core.md)
- [`flight.core.storage`](storage.md)
- [`flight.core.composition`](composition.md)
