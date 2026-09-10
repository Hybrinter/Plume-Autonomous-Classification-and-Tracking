# flight.payload.gimbal.outer

**Source:** `packages/flight/src/flight/payload/gimbal/outer.py`
**Kind:** pure module

## Purpose

The outer law forms the rate reference `r` from co-rotating feedforward, residual
rate, and proportional elevation error. TRACKING and REWIND use hardware and
science-boundary rate limits. Smear is a separate science-quality estimate.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `smear_cap_rad_s` | function | Optical rate threshold used for science qualification |
| `clip_rate` | function | Symmetric rate clip |
| `outer_rate` | function | Mode-dependent rate reference |

## Inputs and outputs

`outer_rate` takes predictor and residual rates, `e_hat`, `K_p`, arbiter mode, a
live-target flag, elevation, science max, hardware slew, live exposure, smear
budget, band IFOV, and science min. It returns `r` in rad/s.

## Behavior

1. Compute image-smear estimates separately for science-frame qualification.
2. In REWIND, drive toward the science limb at the hardware cap. At the limb,
   `r` is 0.
3. In TRACKING with a live aggregate, form
   `omega_t_nom + omega_t_res + K_p * e_hat` and clip to the hardware rate. Zero `r` that would leave
   `[theta_sci_min, theta_sci_max]`. Navigation is optional: without an ISS
   sample, `omega_t_nom` is zero and visual residual feedback remains active.
4. Otherwise return `0.0` (limb wait, cold TRACKING, or unused SAFE path).

For motion toward either science boundary, a stopping-distance governor also
limits the commanded rate by both `sqrt(2 * tau_max/J * remaining_angle)` and
`inner_kp * remaining_angle`. The latter accounts for the nominal rate-loop
response. Independent containment of passive or failed-drive motion remains a
hardware requirement.

The inner loop applies the same governor against a configurable
`science_boundary_guard_deg` inside each boundary. Its default 0.25° is a
provisional calibration value to absorb rate-loop and feedback error.

Exposure and IFOV remain inputs during migration to explicit science-quality
telemetry, but they do not reduce gimbal control authority.

## Errors and faults

None.

## Messages

None.

## Configuration

`OuterLoopConfig.Kp` sets the proportional gain. Hardware slew, torque, and
inertia come from `GimbalConfig`; the inner bandwidth comes from `InnerLoopConfig`.

## Constraints

SAFE / STOW / HOME are outside this tracking-rate law. The function is pure.

## Related documents

- [`flight.payload.gimbal.predictor`](predictor.md)
- [`flight.payload.tracking.residual`](../tracking/residual.md)
- [`flight.payload.control`](../control.md)
