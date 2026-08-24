# flight.hal.drivers_real.gimbal

**Source:** `packages/flight/src/flight/hal/drivers_real/gimbal.py`
**Kind:** driver

## Purpose

Drives a two-axis serial PTU gimbal over an ASCII line protocol. The driver satisfies
`GimbalActuator` structurally. It imports `pyserial` inside `__init__`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RealGimbal` | class | Serial PTU driver |

## Inputs and outputs

Constructor:

- `clock` (`Clock`): timestamps encoder reads
- `cfg` (`GimbalConfig | None`): travel limits, poses, and serial link settings
- `timeout_s` (float): serial read timeout, default 1.0 s

Raises `ImportError` when `pyserial` is absent. Raises `ValueError` when
`cfg.serial_port` is empty.

Protocol methods match `GimbalActuator`. See
[`flight.hal.interfaces.gimbal`](../interfaces/gimbal.md).

## Behavior

1. The constructor opens the configured serial port at `cfg.serial_baud`.
2. Each transaction writes one command line and reads one response line.
3. A `*` response prefix means success. Any other prefix or a serial I/O error maps to
   `GIMBAL_FAULT`.
4. `goto_angle` clamps azimuth and elevation to travel limits, then sends `PP` and `TP`
   commands with signed encoder counts.
5. `set_rate` clamps rates to `max_hw_slew_rate_deg_per_s`, then sends `PS` and `TS`
   commands.
6. `home` and `stow` call `goto_angle` with the configured home or stow pose.
7. `read_position` queries bare `PP` and `TP`, converts counts to degrees, and stamps
   the pose with `clock.monotonic_s()`.
8. `read_stow_switch` reads the encoder pose and returns `True` when both axes are
   within 0.5 deg of the stow pose. The reference PTU has no discrete stow switch.
9. A threading lock serializes all serial transactions.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `FaultCode.GIMBAL_FAULT` | Non-`*` response, serial I/O error, or unparseable position line |

## Messages

None.

## Configuration

Reads `GimbalConfig` fields:

- Travel limits: `az_min_deg`, `az_max_deg`, `el_min_deg`, `el_max_deg`
- Poses: `home_az_deg`, `home_el_deg`, `stow_az_deg`, `stow_el_deg`
- Slew: `max_hw_slew_rate_deg_per_s`
- Serial: `serial_port`, `serial_baud`, `counts_per_deg`

## Constraints

- Angle-to-count conversion uses `counts_per_deg`.
- The verb set (`PP`, `TP`, `PS`, `TS`) is a reference assumption pending HIL validation.
- Construction requires a non-empty `serial_port`.

## Related documents

- [`flight.hal.drivers_real`](../drivers_real.md)
- [`flight.hal.interfaces.gimbal`](../interfaces/gimbal.md)
