# flight.hal.drivers_sim.gimbal

**Source:** `packages/flight/src/flight/hal/drivers_sim/gimbal.py`
**Kind:** driver

## Purpose

`SimGimbal` integrates `J * omega_dot + B * omega + friction = tau` in SI, where
friction is a Coulomb/static-friction term (zero drive torque below the configured
breakaway magnitude). It quantizes the encoder at the real XD-C controller's count
resolution, adds seeded Gaussian noise sized to the datasheet effective resolution,
and satisfies `GimbalActuator` structurally for SIL and tests.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SimGimbal` | class | Rigid-body elevation plant |
| `GimbalSnapshot` | class | Cached delivered encoder and plant truth |

## Inputs and outputs

Construction takes a `Clock`, optional `GimbalConfig`, optional initial elevation,
and the inner period used for frozen-clock catch-up.

| Method | Inputs | Outputs |
| --- | --- | --- |
| `set_torque(tau_nm)` | Torque in N·m | `Ok(None)` |
| `goto_angle(el_deg)` | Target degrees | `Ok(None)` |
| `home()` | None | `Ok(None)` |
| `stow()` | None | `Ok(None)` |
| `read_position()` | None | `Result[GimbalPosition, FaultCode]` |
| `advance_plant()` | None | None (sim-only plant integrate) |
| `snapshot()` | None | `GimbalSnapshot` (sim-only, non-mutating) |
| `read_stow_switch()` | None | `Result[bool, FaultCode]` |
| `freeze_encoder()` | None | Hold encoder reads at the current pose |

Observability properties: `true_el_deg`, `true_omega_rad_s`. `advance_plant` and
`snapshot` are sim-only and are not on `GimbalActuator`.

## Behavior

1. `set_torque` clips torque and integrates elapsed clock time, minus catch-up debt.
2. Repeated `set_torque` at a frozen clock steps one inner period per call and
   records catch-up debt so a later clock jump does not double-count.
3. `stow` / `home` / `goto_angle` latch a pose target. Motion comes from torque.
   The driver does not close a position or rate loop.
4. `read_position` quantizes true elevation to encoder counts and adds Gaussian
   noise. It caches the delivered elevation and last-feedback time.
5. `advance_plant` integrates the plant to the clock. It does not sample the
   encoder or update last-feedback time.
6. `snapshot` returns the cached delivered encoder, true pose and rate, held
   torque, and stow state. It does not draw noise, integrate the plant, expire a
   lease, or write last-feedback time. Stow switch uses the current true pose.
7. `read_stow_switch` is true after `stow()` and when elevation is within 0.5 deg of
   the stow pose.
8. Coulomb/static friction and travel/slew clips apply inside the ODE step.

## Errors and faults

None under normal operation. The sim driver does not return `Err` on commands or
reads.

## Messages

None.

## Configuration

Reads `GimbalConfig` plant scalars (including `tau_coulomb_nm`), travel, slew, encoder
counts, noise, and seed.

## Constraints

- The payload catch-up methods step the plant at frozen clock time. The harness
  advances `ManualClock` after the step.
- The driver enforces the hardware envelope from config.
- `snapshot` is an observer. It is not an encoder acquisition.

## Related documents

- [`flight.hal.interfaces.gimbal`](../interfaces/gimbal.md)
- [`flight.hal.drivers_sim`](../drivers_sim.md)
- [`flight.hal.drivers_real.gimbal`](../drivers_real/gimbal.md)
