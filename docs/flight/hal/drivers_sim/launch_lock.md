# flight.hal.drivers_sim.launch_lock

**Source:** `packages/flight/src/flight/hal/drivers_sim/launch_lock.py`
**Kind:** driver

## Purpose

Models the launch-lock pin in memory for SIL and tests. The driver satisfies
`LaunchLock` structurally. It is the only implementation of the Protocol. No real
driver exists.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SimLaunchLock` | class | In-memory launch-lock stand-in |

## Inputs and outputs

Constructor:

- `state` (`LaunchLockState`): initial pin state; default `ENGAGED`

| Method | Inputs | Output |
| --- | --- | --- |
| `release` | none | `Result[None, FaultCode]` |
| `engage` | none | `Result[None, FaultCode]` |
| `read_state` | none | `Result[LaunchLockState, FaultCode]` |

## Behavior

1. The constructor stores the initial modeled state.
2. `release` sets the state to `RELEASED` and returns `Ok(None)`.
3. `engage` sets the state to `ENGAGED` and returns `Ok(None)`.
4. `read_state` returns `Ok` with the current modeled state.

## Errors and faults

None. All methods return `Ok`.

## Messages

None.

## Configuration

None. Initial state is set at construction. `select_drivers` wires this driver for every
profile.

## Constraints

- No real `LaunchLock` driver exists. This is the only implementation.
- Flight startup defaults to `ENGAGED`. SIL defaults to `RELEASED` when
  `sim_inputs.launch_lock_engaged` is `False`.
- State transitions are immediate with no hardware delay model.

## Related documents

- [`flight.hal.drivers_sim`](../drivers_sim.md)
- [`flight.hal.interfaces.launch_lock`](../interfaces/launch_lock.md)
