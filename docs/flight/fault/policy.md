# flight.fault.policy

**Source:** `packages/flight/src/flight/fault/policy.py`
**Kind:** pure module

## Purpose

The policy module maps fault events to SAFE requests for the external mode authority and
gates the fault-owned latch release. It never selects or executes a system mode itself:
SAFE-triggering faults produce a `SystemModeRequestMsg`, and the latch releases only when
an authorized activation record meets the recovery contract.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SAFE_TRIGGERING_FAULTS` | constant | `frozenset` of fault codes that request SAFE mode |
| `enter_safe_request` | function | Builds the SAFE `SystemModeRequestMsg` for a fault code |
| `decide_mode_request` | function | Maps a `FaultEventMsg` to a request or `None` |
| `recovery_authorized` | function | Gates an activation record for latch release |

## Inputs and outputs

- `enter_safe_request(reason, now_iso, request_id)` returns a
  `SystemModeRequestMsg` with `requested_mode=SAFE` and `requested_by="fault"`.
- `decide_mode_request(event, now_iso, request_id)` returns
  `SystemModeRequestMsg | None`.
- `recovery_authorized(activation, *, expected_epoch, last_sequence,
  request_id_consumed, safe_fault_seen_this_tick)` returns a boolean.

## Behavior

1. `decide_mode_request` checks `event.fault_code` against `SAFE_TRIGGERING_FAULTS`.
2. A matching code produces `enter_safe_request`; all other codes produce `None`.
3. `recovery_authorized` returns true only for an authority-approved EXIT_SAFE recovery:
   the record is marked `recovery_authorized`, carries a nonempty unspent `request_id`,
   moves from SAFE to INIT under the expected epoch, is strictly newer than the last
   observed sequence, and no SAFE-triggering fault fired this tick.

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

Log-and-continue codes (no mode request) include `INFERENCE_TIMEOUT`, `STORAGE_FULL`,
`COMM_TIMEOUT`, and command ingress faults (`COMMAND_CRC_FAIL`, `COMMAND_AUTH_FAIL`,
`COMMAND_SEQ_ERROR`, `COMMAND_INVALID`).

## Messages

Builds `SystemModeRequestMsg` values. Does not publish to the bus.

## Configuration

None.

## Constraints

- Pure module with no I/O, bus access, or clock reads. Identities and request IDs arrive
  as explicit arguments.
- SAFE exit requires an authorized recovery activation; there is no automatic recovery.
- `GIMBAL_FAULT` is in the SAFE set when a driver-level gimbal failure may block stowing.

## Related documents

- [`flight.fault`](../fault.md)
- [`flight.fault.app`](app.md)
- [`flight.thermal`](../thermal.md)
- [`flight.electrical`](../electrical.md)
