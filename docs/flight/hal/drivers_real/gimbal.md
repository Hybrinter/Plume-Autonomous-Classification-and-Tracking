# flight.hal.drivers_real.gimbal

**Source:** `packages/flight/src/flight/hal/drivers_real/gimbal.py`
**Kind:** driver

## Purpose

`RealGimbal` is a torque-command stub. The PTU ASCII path is removed. Commands
return `Ok` and do not move hardware. The driver does not import a vendor SDK.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RealGimbal` | class | Torque-command stub satisfying `GimbalActuator` |

## Inputs and outputs

Construction takes a `Clock` and optional `GimbalConfig`.

| Method | Inputs | Outputs |
| --- | --- | --- |
| `set_torque(tau_nm)` | Torque in N·m (ignored) | `Ok(None)` |
| `goto_angle(el_deg)` | Target degrees | `Ok(None)` |
| `home()` | None | `Ok(None)` |
| `stow()` | None | `Ok(None)` |
| `read_position()` | None | `Ok(GimbalPosition)` |
| `read_stow_switch()` | None | `Ok(bool)` |

## Behavior

1. `set_torque` is a no-op `Ok`. Amp current mapping is not implemented.
2. `goto_angle` records a travel-clamped elevation for later reads.
3. `home` and `stow` record the configured poses. `stow` also arms the switch.
4. `read_position` returns the last recorded pose (0 until a pose command).
5. `read_stow_switch` is true when stow was commanded and the recorded pose is near
   stow.

## Errors and faults

None in this stub. All methods return `Ok`.

## Messages

None.

## Configuration

Reads `GimbalConfig` travel limits and stow/home poses.

## Constraints

Construction does not open a serial port. The amp interface is future work.

## Related documents

- [`flight.hal.interfaces.gimbal`](../interfaces/gimbal.md)
- [`flight.hal.drivers_real`](../drivers_real.md)
- [`flight.hal.drivers_sim.gimbal`](../drivers_sim/gimbal.md)
