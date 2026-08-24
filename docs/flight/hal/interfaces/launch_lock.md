# flight.hal.interfaces.launch_lock

**Source:** `packages/flight/src/flight/hal/interfaces/launch_lock.py`
**Kind:** module

## Purpose

Defines the launch-lock pin Protocol. Release and engage are hazardous, ground-commanded
operations. The mechanical app owns this device. No real driver implements this
Protocol today. Only `SimLaunchLock` exists.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `LaunchLock` | class | Runtime-checkable Protocol for the motorized launch-lock pin |

## Inputs and outputs

| Method | Inputs | Output |
| --- | --- | --- |
| `release` | none | `Result[None, FaultCode]` |
| `engage` | none | `Result[None, FaultCode]` |
| `read_state` | none | `Result[LaunchLockState, FaultCode]` |

## Behavior

1. `release` drives the pin to `RELEASED`. The mechanical app gates this call with
   interlocks.
2. `engage` drives the pin to `ENGAGED`. This is a ground-commanded end-of-mission
   operation.
3. `read_state` returns the current mechanism state from the engaged/released
   microswitches.

## Errors and faults

None defined at the Protocol level. `SimLaunchLock` always returns `Ok`.

## Messages

None.

## Configuration

None.

## Constraints

- No real `LaunchLock` driver exists. The device is hardware-deferred.
- SAFE mode does not re-engage the lock autonomously.
- There is no autonomous engage path outside a ground command.

## Related documents

- [`flight.hal.interfaces`](../interfaces.md)
- [`flight.hal.drivers_sim.launch_lock`](../drivers_sim/launch_lock.md)
