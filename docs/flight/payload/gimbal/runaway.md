# flight.payload.gimbal.runaway

**Source:** `packages/flight/src/flight/payload/gimbal/runaway.py`
**Kind:** pure module

## Purpose

This module monitors encoder motion against commanded RATE commands. Sustained divergence
raises `GIMBAL_RUNAWAY` when strikes reach the configured limit.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RunawayState` | dataclass | Last encoder read and strike counter |
| `INITIAL_RUNAWAY_STATE` | constant | Fresh monitor with no prior position |
| `check_runaway` | function | Compares measured rate to commanded rate |

## Inputs and outputs

`check_runaway(state, pos, commanded_az_rate, commanded_el_rate, rate_mode_active,
tolerance_deg_per_s, strike_limit)` returns `(RunawayState, FaultCode | None)`.

## Behavior

1. Reset strikes when `pos` is None.
2. Reset strikes when not in RATE mode, when no prior read exists, or when timestamps
   do not advance.
3. Compute actual az and el rates from consecutive encoder reads.
4. Increment strikes when the rate vector divergence exceeds tolerance; otherwise reset
   to zero.
5. Return `GIMBAL_RUNAWAY` when strikes reach `strike_limit`.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `GIMBAL_RUNAWAY` | Consecutive divergent checks reach `strike_limit` |

## Messages

None.

## Configuration

Uses `ControllerConfig.runaway_rate_tolerance_deg_per_s` and
`runaway_strike_count`. Commanded rates thread from the previous frame in
`ControlState`.

## Constraints

The monitor runs only during active RATE mode. ABSOLUTE, STOW, and HOME profiles reset
the monitor without faulting.

## Related documents

- [`flight.payload.gimbal.safety`](safety.md)
- [`flight.payload.control`](../control.md)
