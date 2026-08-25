# flight.core.health

**Source:** `packages/flight/src/flight/core/health.py`
**Kind:** pure module

## Purpose

The startup health gate checks that every monitored subsystem has emitted at least one
heartbeat before nominal operation.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `missing_heartbeats` | function | Return monitored names not yet seen |
| `startup_healthy` | function | Return true when every monitored name is seen |

## Inputs and outputs

**`missing_heartbeats(seen, monitored) -> set[str]`**

- Inputs: set of subsystem names already seen; tuple of monitored subsystem names.
- Output: monitored names absent from `seen`.

**`startup_healthy(seen, monitored) -> bool`**

- Inputs: same as above.
- Output: `True` when `missing_heartbeats` is empty, else `False`.

## Behavior

1. `missing_heartbeats` computes `set(monitored) - seen`.
2. `startup_healthy` returns `not missing_heartbeats(seen, monitored)`.

`flight.core.main` owns the time-bounded heartbeat collection. It calls `startup_healthy`
in a loop and publishes `ModeChangeMsg(SAFE)` when the window closes with missing
heartbeats.

## Errors and faults

None. The caller publishes `ModeChangeMsg(SAFE)` on failure.

## Messages

None.

## Configuration

None. The caller passes the monitored tuple and the wait window.

## Constraints

- Pure set comparison only. No I/O and no bus access.
- The monitored tuple comes from `MONITORED_SUBSYSTEMS` in composition.

## Related documents

- [`flight.core`](../core.md)
- [`flight.core.main`](main.md)
- [`flight.core.composition`](composition.md)
