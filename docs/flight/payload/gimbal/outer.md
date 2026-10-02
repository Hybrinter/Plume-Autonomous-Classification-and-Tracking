# flight.payload.gimbal.outer

**Source:** `packages/flight/src/flight/payload/gimbal/outer.py`
**Kind:** pure module

## Purpose

The outer rate-law primitives evaluate one `RateDecision` from composed scene
and relative terms. The commanded elevation rate matches the caller's scene
rate and smear-caps leftover along-track image motion.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RateDecision` | dataclass | Scene, requested, commanded rates and limit flags |
| `smear_cap_rad_s` | function | Elevation smear rate cap from live exposure |
| `clip_rate` | function | Symmetric rate clip |
| `stopping_cap` | function | Finite speed cap from remaining angle |
| `boundary_limited` | function | Stopping-distance clip inside the science window |
| `finish` | function | Hardware slew, stopping governor, and science-window guards |
| `rate_decision` | function | `RateDecision` from caller-composed scene and relative terms |

## Inputs and outputs

`rate_decision` takes the scene rate, the relative term, the requested absolute
rate, the smear cap, elevation, science window, hardware slew, and
stopping-governor terms. The caller composes the scene and relative terms; the
function does not take a mode, azimuth rate, or hunt elapsed time.

The return is a `RateDecision`. Callers that command the gimbal read
`commanded_rate_rad_s`.

## Behavior

1. `smear_cap_rad_s` returns `sigma * IFOV / dt_exp` from the live exposure.
   This is the full elevation smear budget.
2. `finish` clips requested rate to hardware slew, then applies the
   science-window stopping governor through `boundary_limited`. Commanded rate
   that would leave `[theta_sci_min, theta_sci_max]` from a bound is 0.
   `hardware_limited` and `science_limited` record those clips. A zero
   remaining angle yields a zero cap before any infinite stopping product.
3. `rate_decision` runs `finish` on the requested rate and returns the
   `RateDecision` with both limit flags.

For motion toward either science boundary, the stopping-distance governor also
limits commanded speed by both `sqrt(2 * tau_max/J * remaining_angle)` and
`inner_kp * remaining_angle`. The latter accounts for the nominal rate-loop
response. Independent containment of passive or failed-drive motion remains a
hardware requirement.

The inner loop applies the same governor against a configurable
`science_boundary_guard_deg` inside each boundary. Its default 0.25 deg is a
provisional calibration value to absorb rate-loop and feedback error.

## Errors and faults

None.

## Messages

None.

## Configuration

`OuterLoopConfig.Kp` sets the proportional gain at the caller. Hardware slew,
torque, and inertia come from `GimbalConfig`; the inner bandwidth comes from
`InnerLoopConfig`. Smear pixels come from
`PreprocessingConfig.max_motion_smear_px` on the upsampled grid. The default of
4 pixels is 0.002636 deg at the default upsampled IFOV.

## Constraints

SAFE / STOW / HOME are outside this rate law. Every function is pure.
`commanded_rate_rad_s` is an absolute gimbal elevation rate, not a
target-relative rate. Production rate mode passes infinite stopping limits.
The detailed plant passes finite deceleration and rate-loop bandwidth.

## Related documents

- [`flight.payload.gimbal.predictor`](predictor.md)
- [`flight.payload.gimbal.scene`](scene.md)
- [`flight.payload.tracking.residual`](../tracking/residual.md)
- [`flight.payload.control`](../control.md)
