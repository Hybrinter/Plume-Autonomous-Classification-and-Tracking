# flight.hal.interfaces.launch_lock

**Source:** `packages/flight/src/flight/hal/interfaces/launch_lock.py`
**Kind:** module

## Purpose

This module defines the `LaunchLock` Protocol for the motorized launch-lock pin. The
mechanical app owns this device. Release is a hazardous ground-commanded operation. SAFE
mode does not re-engage the lock.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `LaunchLock` | Protocol | Launch-lock command and state read surface |

## Inputs and outputs

| Method | Inputs | Outputs |
| --- | --- | --- |
| `release()` | None | `Result[None, FaultCode]` |
| `engage()` | None | `Result[None, FaultCode]` |
| `read_state()` | None | `Result[LaunchLockState, FaultCode]` |

`LaunchLockState` is `ENGAGED`, `RELEASED`, or `UNKNOWN`.

## Behavior

1. The mechanical app calls `release()` after its interlocks pass.
2. Ground commands call `engage()` at end of mission.
3. `read_state()` reports the microswitch-derived mechanism state.
4. SAFE mode does not call `engage()` autonomously.

## Errors and faults

Implementations return `Err(FaultCode)` on actuator or read failures. The Protocol does not
fix the fault code.

## Messages

None.

## Configuration

None at the Protocol level. The sim driver accepts an initial `LaunchLockState` at
construction.

## Constraints

- Only `SimLaunchLock` implements this Protocol today. No real driver exists yet.
- The mechanical app depends on this Protocol only. A future real driver drops in without
  app changes.
- Engage is ground-commanded only. There is no autonomous engage path from SAFE.

## Related documents

- [`flight.hal.interfaces`](interfaces.md)
- [`flight.hal.drivers_sim.launch_lock`](drivers_sim/launch_lock.md)
- [`flight.mechanical`](mechanical.md)
