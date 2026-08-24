# flight.hal.drivers_sim.launch_lock

**Source:** `packages/flight/src/flight/hal/drivers_sim/launch_lock.py`
**Kind:** driver

## Purpose

`SimLaunchLock` models the launch-lock pin in memory for SIL and tests. It satisfies
`LaunchLock` structurally. This is the only launch-lock implementation. No real driver
exists yet.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SimLaunchLock` | class | In-memory launch-lock stand-in |

## Inputs and outputs

Construction takes an optional initial `LaunchLockState` (default `ENGAGED`).

| Method | Inputs | Outputs |
| --- | --- | --- |
| `release()` | None | `Ok(None)` |
| `engage()` | None | `Ok(None)` |
| `read_state()` | None | `Ok(LaunchLockState)` |

## Behavior

1. Construction sets the modeled microswitch state (default `ENGAGED`, the flight
   configuration).
2. `release()` sets state to `RELEASED`.
3. `engage()` sets state to `ENGAGED`.
4. `read_state()` returns the current modeled state.

## Errors and faults

None under normal operation.

## Messages

None.

## Configuration

Initial state comes from the constructor. `select_drivers` maps `SimDriverInputs.launch_lock_engaged`
to `ENGAGED` or `RELEASED`.

## Constraints

- No real `LaunchLock` driver exists yet. Hardware integration is deferred.
- The mechanical app uses the `LaunchLock` Protocol only. This sim driver stands in until a
  real driver is added.
- Commands always succeed. No actuator fault simulation is modeled.

## Related documents

- [`flight.hal.interfaces.launch_lock`](interfaces/launch_lock.md)
- [`flight.hal.drivers_sim`](drivers_sim.md)
- [`flight.mechanical`](mechanical.md)
