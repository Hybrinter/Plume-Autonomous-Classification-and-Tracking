# flight.hal.drivers_sim.gimbal

**Source:** `packages/flight/src/flight/hal/drivers_sim/gimbal.py`
**Kind:** driver

## Purpose

Models a two-axis gimbal with first-order dynamics for SIL and tests. The driver
satisfies `GimbalActuator` structurally. Position integrates lazily on every public
call using the injected clock.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SimGimbal` | class | First-order gimbal dynamics model |

## Inputs and outputs

Constructor:

- `clock` (`Clock`): elapsed-time source for lazy integration
- `cfg` (`GimbalConfig | None`): travel limits, poses, dynamics, and noise settings
- `az_deg` (float): initial azimuth, default 0.0
- `el_deg` (float): initial elevation, default 0.0

Protocol methods match `GimbalActuator`. See
[`flight.hal.interfaces.gimbal`](../interfaces/gimbal.md).

## Behavior

1. Every public method calls `_integrate()` first to advance the pose by elapsed clock
   time since the previous call.
2. In `RATE` mode the pose integrates clamped commanded rates, limited per step by the
   hardware slew envelope.
3. In `ABSOLUTE`, `STOW`, and `HOME` modes the pose moves toward the target with a
   first-order exponential step, clamped per step by the slew envelope.
4. After each integration step the pose clamps to travel limits.
5. `goto_angle` sets absolute targets and enters `ABSOLUTE` mode.
6. `set_rate` sets axis rates and enters `RATE` mode.
7. `home` and `stow` set the configured pose targets and enter `HOME` or `STOW` mode.
8. `stow` also sets an internal flag that arms stow-switch logic.
9. `read_position` adds seeded Gaussian encoder noise and returns a `GimbalPosition`
   stamped with the integration timestamp.
10. `read_stow_switch` returns `True` only after `stow` was commanded and both axes are
    within 0.5 deg of the stow pose.
11. All command methods return `Ok(None)`. The sim never fails hardware commands.
12. Repeated calls at the same clock time are a no-op for integration (`dt <= 0`).

## Errors and faults

None. All methods return `Ok`.

## Messages

None.

## Configuration

Reads `GimbalConfig` fields:

- Travel limits and poses (same as the real driver)
- `max_hw_slew_rate_deg_per_s`
- `sim_time_constant_s`, `sim_encoder_noise_deg`, `sim_seed`

## Constraints

- The SIL harness must advance the clock between steps or the pose does not move.
- Stow-switch tolerance is 0.5 deg on both axes.
- Encoder noise uses a seeded `numpy` generator from `sim_seed`.

## Related documents

- [`flight.hal.drivers_sim`](../drivers_sim.md)
- [`flight.hal.interfaces.gimbal`](../interfaces/gimbal.md)
