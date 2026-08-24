# flight.payload.gimbal.runaway

**Source:** `packages/flight/src/flight/payload/gimbal/runaway.py`
**Kind:** pure module

## Purpose

The runaway monitor compares commanded RATE against measured encoder motion. Sustained
divergence raises `GIMBAL_RUNAWAY`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RunawayState` | class | Last encoder read and strike counter |
| `INITIAL_RUNAWAY_STATE` | constant | Empty monitor state |
| `check_runaway` | function | One divergence check per frame |

## Inputs and outputs

`check_runaway(state, pos, commanded_az_rate_deg_per_s, commanded_el_rate_deg_per_s,
rate_mode_active, tolerance_deg_per_s, strike_limit)` returns `(RunawayState, FaultCode | None)`.

## Behavior

1. When `pos` is None, reset strikes and clear the last position.
2. When not in RATE mode, no prior read exists, or timestamps do not advance, reset strikes and
   store the new position.
3. Compute actual az and el rates from encoder delta over elapsed time.
4. Compare the hypotenuse of rate error to `tolerance_deg_per_s`.
5. Increment strikes on divergence; clear strikes on agreement.
6. Return `GIMBAL_RUNAWAY` when strikes reach `strike_limit`.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `GIMBAL_RUNAWAY` | Consecutive divergent checks reach `strike_limit` |

## Messages

None.

## Configuration

Reads `ControllerConfig.runaway_rate_tolerance_deg_per_s` and `runaway_strike_count`.

## Constraints

Pure module. `rate_mode_active` is true when prior commanded rates were non-zero. ABSOLUTE,
STOW, and HOME paths reset the monitor.

## Related documents

- [`flight.payload.gimbal`](gimbal.md)
- [`flight.payload.control`](control.md)
