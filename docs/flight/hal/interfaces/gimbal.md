# flight.hal.interfaces.gimbal

**Source:** `packages/flight/src/flight/hal/interfaces/gimbal.py`
**Kind:** module

## Purpose

Defines the gimbal actuator Protocol and the timestamped encoder position type. The
closed-loop surface covers absolute angle, rate, home, stow, encoder readback, and stow
switch sensing.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `GimbalPosition` | class | Frozen dataclass with azimuth, elevation, and encoder timestamp |
| `GimbalActuator` | class | Runtime-checkable Protocol for the pointing gimbal |

## Inputs and outputs

`GimbalPosition` fields:

- `az_deg` (float): Azimuth in degrees. Positive is right of boresight.
- `el_deg` (float): Elevation in degrees. Positive is above boresight.
- `timestamp_s` (float): Monotonic seconds at the encoder read.

`GimbalActuator` methods:

| Method | Inputs | Output |
| --- | --- | --- |
| `goto_angle` | `az_deg`, `el_deg` | `Result[None, FaultCode]` |
| `set_rate` | `az_rate_deg_per_s`, `el_rate_deg_per_s` | `Result[None, FaultCode]` |
| `home` | none | `Result[None, FaultCode]` |
| `stow` | none | `Result[None, FaultCode]` |
| `read_position` | none | `Result[GimbalPosition, FaultCode]` |
| `read_stow_switch` | none | `Result[bool, FaultCode]` |

## Behavior

1. `goto_angle` commands an absolute pointing. The driver clamps to travel limits.
2. `set_rate` commands axis rates. The driver clamps to the hardware slew envelope.
3. `home` drives to the configured home pose.
4. `stow` drives to the configured stow pose.
5. `read_position` returns timestamped encoder angles.
6. `read_stow_switch` returns `True` when the gimbal is mechanically at the stow pose.

## Errors and faults

Each method may return `Err(FaultCode.GIMBAL_FAULT)` on a driver-level failure. The
Protocol does not define the trigger. See the concrete driver pages.

## Messages

None.

## Configuration

None. Drivers read `GimbalConfig` at construction.

## Constraints

- The driver enforces the hardware travel and slew envelope.
- The arbiter enforces the mission envelope separately.
- `read_stow_switch` returns a boolean, not a fault on a closed switch.

## Related documents

- [`flight.hal.interfaces`](../interfaces.md)
- [`flight.hal.drivers_real.gimbal`](../drivers_real/gimbal.md)
- [`flight.hal.drivers_sim.gimbal`](../drivers_sim/gimbal.md)
