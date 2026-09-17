# flight.payload.gimbal.homing

**Source:** `packages/flight/src/flight/payload/gimbal/homing.py`
**Kind:** pure module

## Purpose

The module runs the placeholder INIT assumed-datum creep and the STOW rest slew.
There is no vendor index search.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `HomingPhase` | enum | `CREEP_OUT`, `CREEP_BACK`, `DATUM`, `COMPLETE`, `FAILED` |
| `HomingState` | dataclass | Current phase and phase start time |
| `HomingTick` | dataclass | Next state, rate, failed flag, complete flag |
| `initial_homing` | function | Starts `CREEP_OUT` at `now` |
| `step_homing` | function | Advances INIT homing by one outer tick |
| `stow_rate_deg_per_s` | function | Rate toward rest and arrival flag |

## Inputs and outputs

`initial_homing(now)` returns `HomingState(phase=CREEP_OUT, phase_started_s=now)`.

`step_homing(state, now, el_deg, cfg)` returns a `HomingTick`.

`stow_rate_deg_per_s(el_deg, cfg)` returns `(rate_deg_per_s, arrived)`.

## Behavior

1. An elevation outside `[stow_el_deg - 1, el_science_min_deg + 1]` fails INIT.
2. `CREEP_OUT` commands `+init_creep_rate_deg_per_s` until span time or the
   science-window edge at `0 deg`.
3. `CREEP_BACK` commands the negative rate until rest or span plus press
   timeout.
4. `DATUM` commands rate `0` and marks `COMPLETE` on the next tick.
5. STOW commands the creep rate toward `stow_el_deg`. Arrival uses
   `stow_arrive_tol_deg`.

## Errors and faults

The module does not raise. Envelope trip sets `HomingPhase.FAILED`. The payload
app publishes `GIMBAL_RUNAWAY` for that case.

## Messages

None. The payload app publishes `ModeRequestMsg` after complete or failed ticks.

## Configuration

Reads `GimbalConfig.init_creep_rate_deg_per_s`, `init_creep_span_deg`,
`init_press_timeout_s`, `stow_el_deg`, `el_science_min_deg`, and
`stow_arrive_tol_deg`.

## Constraints

- Pure module with no I/O, bus access, or clock reads.
- Time arrives as `now` from the caller.
- The function does not call a vendor `home()` or index search.

## Related documents

- [`flight.payload.gimbal`](../gimbal.md)
- [`flight.payload.app`](../../app.md)
- [`flight.libs.config`](../../../libs/config.md)
