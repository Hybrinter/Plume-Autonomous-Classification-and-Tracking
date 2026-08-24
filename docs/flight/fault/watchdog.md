# flight.fault.watchdog

**Source:** `packages/flight/src/flight/fault/watchdog.py`
**Kind:** pure module

## Purpose

The watchdog module detects silent subsystem death through missed heartbeats. It is a pure module:
no clock reads, no bus access, no I/O.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `WatchdogEntry` | class | Frozen per-subsystem watchdog record |
| `build_entries` | function | Constructs the starting entries dict for monitored subsystems |
| `check_heartbeats` | function | Increments miss counts and emits `WATCHDOG_EXPIRE` at the threshold |

## Inputs and outputs

- `build_entries(subsystems, max_interval_s, now)` returns a `dict[str, WatchdogEntry]`.
- `check_heartbeats(entries, now, max_miss_count, now_iso)` returns
  `(updated_entries, list[FaultEventMsg])`.

## Behavior

1. `build_entries` sets `last_heartbeat_time=now` and `miss_count=0` for each monitored subsystem.
2. On each `check_heartbeats` call, compare `now - last_heartbeat_time` to `max_interval_s`.
3. When elapsed time exceeds the interval, increment `miss_count`.
4. When `miss_count` reaches `max_miss_count`, append a `FaultEventMsg(WATCHDOG_EXPIRE)`.
5. Return the updated entries dict and any emitted faults.

## Errors and faults

Emits `FaultCode.WATCHDOG_EXPIRE` with the subsystem name in the fault detail when the miss
threshold is reached. Does not remove the entry after emission; misses keep accumulating until a
heartbeat resets `miss_count` to zero.

## Messages

Produces `FaultEventMsg` values in the returned fault list. Does not publish to the bus directly.

## Configuration

None. The caller passes `max_interval_s` and `max_miss_count`.

## Constraints

- Does not mutate the input entries dict in place; returns a new dict.
- Timestamps on emitted faults use the injected `now_iso` string.
- Elapsed timing uses monotonic seconds from the caller.

## Related documents

- [`flight.fault`](../fault.md)
- [`flight.fault.app`](app.md)
- [`flight.fault.policy`](policy.md)
