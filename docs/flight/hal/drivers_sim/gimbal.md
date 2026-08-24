# flight.hal.drivers_sim.gimbal

**Source:** `packages/flight/src/flight/hal/drivers_sim/gimbal.py`
**Kind:** driver

## Purpose

`SimGimbal` models a two-axis gimbal with first-order dynamics, travel and slew limits, and
seeded encoder noise. It satisfies `GimbalActuator` structurally for SIL and tests.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SimGimbal` | class | First-order gimbal dynamics driver |

## Inputs and outputs

Construction takes a `Clock`, an optional `GimbalConfig`, and optional initial azimuth and
elevation in degrees.

| Method | Inputs | Outputs |
| --- | --- | --- |
| `goto_angle(az_deg, el_deg)` | Target degrees | `Ok(None)` |
| `set_rate(az_rate_deg_per_s, el_rate_deg_per_s)` | Axis rates | `Ok(None)` |
| `home()` | None | `Ok(None)` |
| `stow()` | None | `Ok(None)` |
| `read_position()` | None | `Result[GimbalPosition, FaultCode]` |
| `read_stow_switch()` | None | `Result[bool, FaultCode]` |

## Behavior

1. Every public call runs lazy integration first. Integration advances pose by elapsed
   monotonic clock time since the previous call.
2. In rate mode, the driver integrates clamped commanded rates with a per-step slew cap.
3. In absolute, home, and stow modes, the driver moves toward the target with a first-order
   exponential step, also capped by the slew envelope.
4. After integration, the driver clamps pose to configured travel limits.
5. `read_position()` adds Gaussian encoder noise from a seeded RNG and returns a
   timestamped pose.
6. `read_stow_switch()` returns `True` only after `stow()` was called and both axes are
   within 0.5 deg of the stow pose.
7. Sim hardware commands never fail. All command methods return `Ok(None)`.

## Errors and faults

None under normal operation. The sim driver does not return `Err` on commands or reads.

## Messages

None.

## Configuration

Reads `GimbalConfig`: travel limits, stow and home poses, max hardware slew,
`sim_time_constant_s`, `sim_encoder_noise_deg`, and `sim_seed`.

## Constraints

- The injected clock must advance between SIL steps or the pose does not move.
- Repeated calls at the same clock time are idempotent (`dt <= 0` is a no-op).
- The driver enforces the hardware envelope from config. The arbiter enforces mission limits
  above it.

## Related documents

- [`flight.hal.interfaces.gimbal`](interfaces/gimbal.md)
- [`flight.hal.drivers_sim`](drivers_sim.md)
- [`flight.hal.drivers_real.gimbal`](drivers_real/gimbal.md)
