# flight.hal.interfaces.gimbal

**Source:** `packages/flight/src/flight/hal/interfaces/gimbal.py`
**Kind:** module

## Purpose

This module defines the closed-loop gimbal surface. `GimbalActuator` covers absolute
angle, rate, home, stow, encoder readback, and stow-switch sensing. `GimbalPosition`
carries a monotonic encoder timestamp.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `GimbalPosition` | dataclass | Timestamped azimuth and elevation in degrees |
| `GimbalActuator` | Protocol | Closed-loop gimbal command and readback surface |

## Inputs and outputs

| Method | Inputs | Outputs |
| --- | --- | --- |
| `goto_angle(az_deg, el_deg)` | Target azimuth and elevation in degrees | `Result[None, FaultCode]` |
| `set_rate(az_rate_deg_per_s, el_rate_deg_per_s)` | Axis rates in deg/s | `Result[None, FaultCode]` |
| `home()` | None | `Result[None, FaultCode]` |
| `stow()` | None | `Result[None, FaultCode]` |
| `read_position()` | None | `Result[GimbalPosition, FaultCode]` |
| `read_stow_switch()` | None | `Result[bool, FaultCode]` |

## Behavior

1. The payload gimbal path issues pointing commands through the typed methods.
2. The driver clamps commands to the hardware travel and slew envelope.
3. `read_position()` returns encoder angles with a monotonic timestamp.
4. `read_stow_switch()` returns `True` when the mechanism is at the stow pose.
5. The gimbal arbiter enforces mission limits above the driver envelope.

## Errors and faults

Driver implementations map hardware and serial failures to `GIMBAL_FAULT` or related
codes. The Protocol itself does not fix fault values.

## Messages

None.

## Configuration

None at the Protocol level. Concrete drivers read poses, limits, and link settings from
`GimbalConfig`.

## Constraints

- The legacy delta-command path is removed. Actuation flows through the typed methods.
- The driver enforces the hardware envelope. The arbiter enforces the mission envelope.
- `GimbalPosition.timestamp_s` uses the injected `Clock` monotonic time.

## Related documents

- [`flight.hal.interfaces`](interfaces.md)
- [`flight.hal.drivers_real.gimbal`](drivers_real/gimbal.md)
- [`flight.hal.drivers_sim.gimbal`](drivers_sim/gimbal.md)
- [`flight.payload.gimbal`](payload/gimbal.md)
