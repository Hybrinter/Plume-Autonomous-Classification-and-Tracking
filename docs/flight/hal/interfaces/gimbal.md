# flight.hal.interfaces.gimbal

**Source:** `packages/flight/src/flight/hal/interfaces/gimbal.py`
**Kind:** module

## Purpose

This module defines the elevation gimbal surface. `GimbalActuator` commands torque
and reads encoder elevation. `stow` / `home` / `goto_angle` latch a pose target for
stow-switch arming. They do not run a position or rate controller. There is no
azimuth axis and no rate command.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `GimbalPosition` | dataclass | Timestamped elevation in degrees |
| `GimbalActuator` | Protocol | Torque, pose targets, encoder, and stow switch |

## Inputs and outputs

| Method | Inputs | Outputs |
| --- | --- | --- |
| `set_torque(tau_nm)` | Torque in N·m | `Result[None, FaultCode]` |
| `goto_angle(el_deg)` | Target elevation in degrees | `Result[None, FaultCode]` |
| `home()` | None | `Result[None, FaultCode]` |
| `stow()` | None | `Result[None, FaultCode]` |
| `read_position()` | None | `Result[GimbalPosition, FaultCode]` |
| `read_stow_switch()` | None | `Result[bool, FaultCode]` |

## Behavior

1. Tracking and STOW / HOME / GOTO write torque through `set_torque`.
2. `stow`, `home`, and `goto_angle` latch a pose target and arm stow-switch logic.
   The payload position loop turns that target into a rate reference `r` for the
   inner PI. The driver does not close a position or rate loop.
3. The driver clips torque, rate, and travel to the hardware envelope.
4. `read_position()` returns encoder elevation with a monotonic timestamp.
5. `read_stow_switch()` returns `True` when the mechanism is at the stow pose.

## Errors and faults

Driver implementations map hardware failures to `GIMBAL_FAULT` or related codes.
The Protocol itself does not fix fault values.

## Messages

None.

## Configuration

None at the Protocol level. Concrete drivers read poses, limits, and plant scalars
from `GimbalConfig`.

## Constraints

`GimbalPosition` has elevation and timestamp only. `GimbalPosition.timestamp_s` uses
the injected `Clock` monotonic time.

## Related documents

- [`flight.hal.interfaces`](../interfaces.md)
- [`flight.hal.drivers_real.gimbal`](../drivers_real/gimbal.md)
- [`flight.hal.drivers_sim.gimbal`](../drivers_sim/gimbal.md)
- [`flight.payload.gimbal`](../../payload/gimbal.md)
