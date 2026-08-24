# flight.fault.policy

**Source:** `packages/flight/src/flight/fault/policy.py`
**Kind:** pure module

## Purpose

This module maps fault events to mode-change requests. It defines which fault codes trigger
SAFE mode and builds the corresponding `ModeChangeMsg` values.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SAFE_TRIGGERING_FAULTS` | constant | Frozenset of fault codes that request SAFE mode |
| `enter_safe_mode` | function | Builds a `ModeChangeMsg` with `new_mode=SAFE` |
| `exit_safe_mode` | function | Builds a `ModeChangeMsg` with `new_mode=IDLE` |
| `can_exit_safe` | function | Returns whether an EXIT_SAFE command may un-latch SAFE |
| `decide_mode_change` | function | Maps a `FaultEventMsg` to a mode change or `None` |

## Inputs and outputs

`decide_mode_change(event, now_iso)` returns a `ModeChangeMsg(SAFE)` when
`event.fault_code` is in `SAFE_TRIGGERING_FAULTS`, else `None`.

`enter_safe_mode(reason, now_iso)` and `exit_safe_mode(cleared_by, now_iso)` return a
`ModeChangeMsg` with the requested mode and a `requested_by` string.

`can_exit_safe(safe_latched, safe_fault_seen_this_tick)` returns `True` only when SAFE is
latched and no SAFE-triggering fault fired in the current tick.

## Behavior

1. Test `event.fault_code` against `SAFE_TRIGGERING_FAULTS`.
2. Return `enter_safe_mode` when the code is a member; return `None` otherwise.

SAFE-triggering codes are `INFERENCE_NAN`, `CAMERA_STALL`, `THERMAL_OVER_LIMIT`,
`POWER_OVER_LIMIT`, `GIMBAL_RUNAWAY`, `GIMBAL_FAULT`, `WATCHDOG_EXPIRE`, `MODEL_CORRUPT`,
and `PROCESS_DIED`.

Log-and-continue codes include `NONE`, `INFERENCE_TIMEOUT`, `STORAGE_FULL`, `COMM_TIMEOUT`,
`COMMAND_CRC_FAIL`, `COMMAND_AUTH_FAIL`, `COMMAND_SEQ_ERROR`, and `COMMAND_INVALID`.

## Errors and faults

None. The module returns messages or booleans; it does not raise or return `Err`.

## Messages

None.

## Configuration

None.

## Constraints

The module is pure. It performs no I/O, reads no clock, and touches no bus. Command-ingress
fault codes never trigger SAFE mode.

## Related documents

- [`flight.fault`](../fault.md)
- [`flight.fault.app`](app.md)
