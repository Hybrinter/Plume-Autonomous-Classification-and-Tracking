# flight.fault.policy

**Source:** `packages/flight/src/flight/fault/policy.py`
**Kind:** pure module

## Purpose

The policy module maps fault events to mode-change requests. It defines which fault codes trigger
SAFE entry and builds the corresponding `ModeChangeMsg` values.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SAFE_TRIGGERING_FAULTS` | constant | `frozenset` of fault codes that request SAFE mode |
| `enter_safe_mode` | function | Builds `ModeChangeMsg(SAFE)` for a given fault code |
| `exit_safe_mode` | function | Builds `ModeChangeMsg(IDLE)` after ground clearance |
| `can_exit_safe` | function | Returns whether an `EXIT_SAFE` command may un-latch SAFE |
| `decide_mode_change` | function | Maps a `FaultEventMsg` to a mode change or `None` |

## Inputs and outputs

- `enter_safe_mode(reason, now_iso)` returns a `ModeChangeMsg` with `new_mode=SAFE`.
- `exit_safe_mode(cleared_by, now_iso)` returns a `ModeChangeMsg` with `new_mode=IDLE`.
- `can_exit_safe(safe_latched, safe_fault_seen_this_tick)` returns a boolean.
- `decide_mode_change(event, now_iso)` returns `ModeChangeMsg | None`.

## Behavior

1. `decide_mode_change` checks `event.fault_code` against `SAFE_TRIGGERING_FAULTS`.
2. A matching code produces `enter_safe_mode`; all other codes produce `None`.
3. `can_exit_safe` returns true when SAFE is latched and no SAFE-triggering fault fired in the
   current tick.

## Errors and faults

`SAFE_TRIGGERING_FAULTS` contains:

- `INFERENCE_NAN`
- `CAMERA_STALL`
- `THERMAL_OVER_LIMIT`
- `POWER_OVER_LIMIT`
- `GIMBAL_RUNAWAY`
- `GIMBAL_FAULT`
- `WATCHDOG_EXPIRE`
- `MODEL_CORRUPT`
- `PROCESS_DIED`

Log-and-continue codes (no mode change) include `INFERENCE_TIMEOUT`, `STORAGE_FULL`,
`COMM_TIMEOUT`, and command ingress faults (`COMMAND_CRC_FAIL`, `COMMAND_AUTH_FAIL`,
`COMMAND_SEQ_ERROR`, `COMMAND_INVALID`).

## Messages

Builds `ModeChangeMsg` values. Does not publish to the bus.

## Configuration

None.

## Constraints

- Pure module with no I/O, bus access, or clock reads.
- SAFE exit requires an explicit ground command; there is no automatic recovery.
- `GIMBAL_FAULT` is in the SAFE set when a driver-level gimbal failure may block stowing.

## Related documents

- [`flight.fault`](../fault.md)
- [`flight.fault.app`](app.md)
- [`flight.thermal`](../thermal.md)
- [`flight.electrical`](../electrical.md)
