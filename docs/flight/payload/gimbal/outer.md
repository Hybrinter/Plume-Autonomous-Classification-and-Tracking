# flight.payload.gimbal.outer

**Source:** `packages/flight/src/flight/payload/gimbal/outer.py`
**Kind:** pure module

## Purpose

The tracking rate law forms an absolute elevation rate `r` for the production
`set_rate` path and the detailed-plant inner PI. It matches the elevation scene
rate and smear-caps only leftover along-track image motion.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `smear_cap_rad_s` | function | Elevation-relative smear |ω| cap from live exposure |
| `clip_rate` | function | Symmetric rate clip |
| `outer_rate` | function | Mode-dependent absolute rate reference |

## Inputs and outputs

`outer_rate` takes predictor elevation rate, residual rate, `e_hat`, `K_p`, arbiter
mode, a live-target flag, elevation, science window, hardware slew, live exposure,
smear budget, band IFOV, stopping-governor terms, REWIND elapsed time, REWIND
sharp-window duration, and diagnostic `omega_az`. It returns `r` in rad/s.

## Behavior

1. `omega_sharp,el` is `σ * IFOV / Δt_exp` from the live exposure. Unactuated
   `omega_az` does not reduce that cap.
2. In REWIND, if elevation is at the science limb, `r` is 0. Inside
   `rewind_sharp_max_s`, `r = omega_t_nom + sign(θ_sci,max − θ_g) * omega_sharp`
   (residual is ignored). After that window, `r` is the hardware cap toward the
   limb.
3. In TRACKING with a live aggregate, `omega_scene = omega_t_nom + omega_t_res`
   is not smear-clipped. Only `K_p * e_hat` is clipped to `±omega_sharp`. Then
   `r = omega_scene + omega_rel`. Zero `r` that would leave
   `[theta_sci_min, theta_sci_max]`. Navigation is optional: without an ISS
   sample, `omega_t_nom` is zero and visual residual feedback remains active.
4. Otherwise return `0.0` (limb wait, cold TRACKING, or unused SAFE path).
5. Hardware slew and the science-window stopping governor clip the absolute `r`
   last.

For motion toward either science boundary, a stopping-distance governor also
limits the commanded rate by both `sqrt(2 * tau_max/J * remaining_angle)` and
`inner_kp * remaining_angle`. The latter accounts for the nominal rate-loop
response. Independent containment of passive or failed-drive motion remains a
hardware requirement.

The inner loop applies the same governor against a configurable
`science_boundary_guard_deg` inside each boundary. Its default 0.25° is a
provisional calibration value to absorb rate-loop and feedback error.

## Errors and faults

None.

## Messages

None.

## Configuration

`OuterLoopConfig.Kp` sets the proportional gain. `OuterLoopConfig.rewind_sharp_max_s`
sets the sharp REWIND window. Hardware slew, torque, and inertia come from
`GimbalConfig`; the inner bandwidth comes from `InnerLoopConfig`. Smear pixels
come from `PreprocessingConfig.max_motion_smear_px`.

## Constraints

SAFE / STOW / HOME are outside this tracking-rate law. The function is pure. `r`
is an absolute gimbal elevation rate, not a target-relative rate.

## Related documents

- [`flight.payload.gimbal.predictor`](predictor.md)
- [`flight.payload.tracking.residual`](../tracking/residual.md)
- [`flight.payload.control`](../control.md)
