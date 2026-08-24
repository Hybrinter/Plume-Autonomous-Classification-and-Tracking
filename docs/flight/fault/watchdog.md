# flight.fault.watchdog

**Source:** `packages/flight/src/flight/fault/watchdog.py`
**Kind:** pure module

## Purpose

This module detects silent subsystem death through missed heartbeats. It tracks per-subsystem
overdue intervals and emits `WATCHDOG_EXPIRE` faults at a configured threshold.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `WatchdogEntry` | class | Frozen record for one monitored subsystem |
| `build_entries` | function | Constructs the starting entries dict |
| `check_heartbeats` | function | Increments miss counts and emits watchdog faults |

## Inputs and outputs

`build_entries(subsystems, max_interval_s, now)` returns a dict mapping each subsystem name
to a fresh `WatchdogEntry`.

`check_heartbeats(entries, now, max_miss_count, now_iso)` returns `(updated_entries, faults)`.
The faults list holds `FaultEventMsg(WATCHDOG_EXPIRE)` values for subsystems at the threshold.

## Behavior

1. For each entry, compute elapsed time since `last_heartbeat_time`.
2. When elapsed exceeds `max_interval_s`, increment `miss_count`.
3. When `miss_count` reaches `max_miss_count`, append a `FaultEventMsg(WATCHDOG_EXPIRE)` for
   that subsystem.
4. Return the updated entries dict and the fault list.

`build_entries` sets `last_heartbeat_time` to `now` and `miss_count` to zero for every
subsystem. Each subsystem gets one full interval before its first miss.

## Errors and faults

The module constructs `FaultEventMsg` values with `FaultCode.WATCHDOG_EXPIRE`. It does not
return `Err`.

## Messages

None. The caller publishes the returned `FaultEventMsg` values on the bus.

## Configuration

None. The caller passes `max_interval_s` and `max_miss_count` as arguments.

## Constraints

The module is pure. It does not read a clock; the caller passes `now` and `now_iso`.
Entries are not mutated in place. After a fault fires, misses keep accumulating until a
heartbeat resets `miss_count` to zero.

## Related documents

- [`flight.fault`](../fault.md)
- [`flight.fault.app`](app.md)
