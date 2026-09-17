# flight.fault.policy

**Source:** `packages/flight/src/flight/fault/policy.py`
**Kind:** pure module

## Purpose

The policy module maps fault events to SAFE-entry requests. It defines which fault codes
trigger SAFE and builds the corresponding `ModeChangeMsg` values.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SAFE_TRIGGERING_FAULTS` | constant | `frozenset` of fault codes that request SAFE mode |
| `enter_safe_mode` | function | Builds `ModeChangeMsg(SAFE)` for a given fault code |
| `enter_init_mode` | function | Builds `ModeChangeMsg(INIT)` after ground clearance |
| `can_enter_init` | function | Returns whether an `ENTER_INIT` command may leave SAFE |
| `decide_mode_change` | function | Maps a `FaultEventMsg` to a mode change or `None` |

## Inputs and outputs

- `enter_safe_mode(reason, now_iso)` returns a `ModeChangeMsg` with `new_mode=SAFE`.
- `enter_init_mode(cleared_by, now_iso)` returns a `ModeChangeMsg` with `new_mode=INIT`.
- `can_enter_init(safe_latched, safe_fault_seen_this_tick)` returns a boolean.
- `decide_mode_change(event, now_iso)` returns `ModeChangeMsg | None`.

## Behavior

1. `decide_mode_change` checks `event.fault_code` against `SAFE_TRIGGERING_FAULTS`.
2. A matching code produces `enter_safe_mode`. All other codes produce `None`.
3. `can_enter_init` returns true when SAFE is latched and no SAFE-triggering fault fired in
   the current tick.

## Errors and faults

`SAFE_TRIGGERING_FAULTS` contains:

- `INFERENCE_NAN`
- `CAMERA_STALL`
- `THERMAL_OVER_LIMIT`
- `POWER_OVER_LIMIT`
- `GIMBAL_RUNAWAY`
- `GIMBAL_FAULT`
- `GIMBAL_ENCODER_INVALID`
- `GIMBAL_CONTROLLER_ERROR`
- `GIMBAL_THERMAL`
- `GIMBAL_SAFETY_TIMEOUT`
- `GIMBAL_CLOSED_LOOP_LOSS`
- `GIMBAL_STALE_FEEDBACK`
- `GIMBAL_TIME_MAPPING`
- `GIMBAL_DUTY_EXHAUSTED`
- `GIMBAL_WATCHDOG_UNCONFIRMED`
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
- SAFE exit requires an explicit ground `ENTER_INIT` command. There is no automatic recovery.
- `GIMBAL_FAULT` is in the SAFE set when a driver-level gimbal failure may block motion.

## Related documents

- [`flight.fault`](../fault.md)
- [`flight.fault.app`](app.md)
- [`flight.fault.mode`](mode.md)
- [`flight.thermal`](../thermal.md)
- [`flight.electrical`](../electrical.md)
