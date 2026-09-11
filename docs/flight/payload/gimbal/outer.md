# flight.payload.gimbal.outer

**Source:** `packages/flight/src/flight/payload/gimbal/outer.py`
**Kind:** pure module

## Purpose

The tracking rate law returns a `RateDecision`. The commanded elevation rate
matches the scene rate and smear-caps leftover along-track image motion.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RateDecision` | dataclass | Scene, requested, commanded rates and limit flags |
| `smear_cap_rad_s` | function | Elevation smear rate cap from live exposure |
| `clip_rate` | function | Symmetric rate clip |
| `stopping_cap` | function | Finite speed cap from remaining angle |
| `boundary_limited` | function | Stopping-distance clip inside the science window |
| `finish` | function | Hardware slew, stopping governor, and science-window guards |
| `outer_rate` | function | Mode-dependent `RateDecision` |

## Inputs and outputs

`outer_rate` takes predictor elevation rate, residual rate, `e_hat`, `K_p`, arbiter
mode, a live-target flag, elevation, science window, hardware slew, live exposure,
smear budget, band IFOV, stopping-governor terms, REWIND elapsed time, and REWIND
sharp-window duration. Residual rate is unused in REWIND. The function does not
take azimuth rate.

The return is a `RateDecision`. Callers that command the gimbal read
`commanded_rate_rad_s`.

## Behavior

1. `omega_sharp,el` is `σ * IFOV / Δt_exp` from the live exposure. This is the
   full elevation smear budget.
2. In REWIND, if elevation is at `theta_sci_max`, commanded rate is 0. Inside
   `rewind_sharp_max_s`, requested relative rate is `+omega_sharp`. After that
   window, requested relative rate is `+omega_hw`. Scene rate is `omega_t_nom`.
   Residual rate is ignored. A negative commanded rate at `theta_sci_min` is 0.
3. In TRACKING with a live aggregate, scene rate is `omega_t_nom + omega_t_res`
   and is not smear-clipped. Only `K_p * e_hat` is clipped to `+-omega_sharp`.
   Requested rate is scene rate plus that relative term. Commanded rate that
   would leave `[theta_sci_min, theta_sci_max]` is 0. Navigation is optional.
   Without an ISS sample, `omega_t_nom` is zero and visual residual feedback
   remains active.
4. Otherwise commanded rate is `0.0` (limb wait, cold TRACKING, or unused SAFE
   path).
5. `finish` clips requested rate to hardware slew, then the science-window
   stopping governor. `hardware_limited` and `science_limited` record those
   clips. A zero remaining angle yields a zero cap before any infinite
   stopping product.

For motion toward either science boundary, the stopping-distance governor also
limits commanded speed by both `sqrt(2 * tau_max/J * remaining_angle)` and
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

SAFE / STOW / HOME are outside this tracking-rate law. The function is pure.
`commanded_rate_rad_s` is an absolute gimbal elevation rate, not a target-relative
rate. Production rate mode passes infinite stopping limits. The detailed plant
passes finite deceleration and rate-loop bandwidth.

## Related documents

- [`flight.payload.gimbal.predictor`](predictor.md)
- [`flight.payload.tracking.residual`](../tracking/residual.md)
- [`flight.payload.control`](../control.md)
