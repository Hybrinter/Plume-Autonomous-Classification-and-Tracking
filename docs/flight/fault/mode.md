# flight.fault.mode

**Source:** `packages/flight/src/flight/fault/mode.py`
**Kind:** pure module

## Purpose

The mode manager is the only source of `ModeChangeMsg`. It maps one frozen
`SystemModeState` and one event to the next state and an optional mode-change
message.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ModeEventKind` | enum | Input discriminant for one step |
| `ModeEvent` | dataclass | Kind, requester, and optional fault code |
| `SystemModeState` | dataclass | Current mode, latch, reason, and suspend flag |
| `initial_mode_state` | function | Boot snapshot: latched `SAFE` |
| `begin_tick` | function | Clears the per-tick SAFE-fault flag |
| `step` | function | Advances the machine by one event |

## Inputs and outputs

`initial_mode_state()` returns `SystemModeState(mode=SAFE, safe_latched=True)`.

`begin_tick(state)` returns a copy with `safe_fault_this_tick=False`.

`step(state, event, now_iso)` returns `(SystemModeState, ModeChangeMsg | None)`.
Illegal edges return the same state and `None`.

## Behavior

1. A SAFE-triggering `FAULT` or `HOMING_FAILED` event moves every powered mode to
   `SAFE` and sets the latch.
2. `ENTER_INIT` is legal only from latched `SAFE` when `safe_fault_this_tick` is
   false. It publishes `INIT` and clears the latch.
3. `HOMING_COMPLETE` is legal only from `INIT`. It publishes `IDLE`.
4. `ENTER_OPERATE` is legal only from `IDLE`. It publishes `OPERATE`.
5. `ENTER_IDLE` is legal only from `OPERATE`. It publishes `IDLE` and clears
   `operate_suspended`.
6. `MODEL_SUSPEND` from `OPERATE` publishes `IDLE` and sets `operate_suspended`.
7. `MODEL_RESUME` returns to `OPERATE` only when `operate_suspended` is true.
8. `ENTER_STOW` is legal only from `IDLE`. It publishes `STOW`.
9. `STOW_COMPLETE` from `STOW` publishes latched `SAFE` with reason `NONE`.

## Errors and faults

The module does not raise. It records `safe_reason` from a triggering
`FaultCode`. `HOMING_FAILED` uses the event fault code.

## Messages

Builds `ModeChangeMsg` values. It does not publish to the bus.

## Configuration

None.

## Constraints

- Pure module with no I/O, bus access, or clock reads.
- Time arrives as `now_iso` from the caller.
- Inner graphs do not publish `ModeChangeMsg`.

## Related documents

- [`flight.fault`](../fault.md)
- [`flight.fault.app`](app.md)
- [`flight.fault.policy`](policy.md)
