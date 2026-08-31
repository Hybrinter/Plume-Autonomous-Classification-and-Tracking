# flight.hal.drivers_real.gimbal

**Source:** `packages/flight/src/flight/hal/drivers_real/gimbal.py`
**Kind:** driver

## Purpose

`RealGimbal` drives a two-axis serial PTU over an ASCII line protocol. It satisfies
`GimbalActuator` structurally. The driver clamps travel and slew before sending encoder
counts. `set_torque` is a stub until the actuator hardware is selected.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RealGimbal` | class | Serial PTU gimbal driver |

## Inputs and outputs

Construction takes a `Clock`, an optional `GimbalConfig`, and a serial read timeout in
seconds.

| Method | Inputs | Outputs |
| --- | --- | --- |
| `goto_angle(az_deg, el_deg)` | Target degrees | `Result[None, FaultCode]` |
| `set_rate(az_rate_deg_per_s, el_rate_deg_per_s)` | Axis rates | `Result[None, FaultCode]` |
| `set_torque(az_nm, el_nm)` | Axis torques in N·m | `Ok(None)` (stub) |
| `home()` | None | `Result[None, FaultCode]` |
| `stow()` | None | `Result[None, FaultCode]` |
| `read_position()` | None | `Result[GimbalPosition, FaultCode]` |
| `read_stow_switch()` | None | `Result[bool, FaultCode]` |

Construction raises `ImportError` when pyserial is absent. It raises `ValueError` when
`GimbalConfig.serial_port` is empty.

## Behavior

1. Construction opens the configured serial port at the configured baud rate.
2. Each command writes one ASCII line (`<verb><signed counts>\n`) and reads one response
   line.
3. A `*` response prefix means success. Any other prefix or I/O error is a fault.
4. `goto_angle` sends `PP` and `TP` with clamped targets. `set_rate` sends `PS` and `TS`
   with clamped rates.
5. `set_torque` returns `Ok(None)` and does not write the serial port. The hardware
   torque map is a stub.
6. `home` and `stow` delegate to `goto_angle` with configured poses.
7. `read_position` queries bare `PP` and `TP`, converts counts to degrees, and stamps
   monotonic time from the clock.
8. `read_stow_switch` infers stow from encoder pose within 0.5 deg of the configured stow
   pose. The reference PTU has no discrete switch.
9. A lock serializes all serial transactions.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `GIMBAL_FAULT` | Non-success PTU response, serial I/O error, or unparseable position line |

## Messages

None.

## Configuration

Reads `GimbalConfig`: travel limits, stow and home poses, max hardware slew, serial port,
baud rate, and `counts_per_deg`.

## Constraints

- pyserial imports inside `__init__` only.
- Verb set (`PP`, `TP`, `PS`, `TS`) is a reference assumption for HIL validation.
- `set_torque` is a stub. It does not command hardware.
- The driver enforces the hardware envelope. The arbiter enforces mission limits above it.

## Related documents

- [`flight.hal.interfaces.gimbal`](interfaces/gimbal.md)
- [`flight.hal.drivers_real`](drivers_real.md)
- [`flight.hal.drivers_sim.gimbal`](drivers_sim/gimbal.md)
