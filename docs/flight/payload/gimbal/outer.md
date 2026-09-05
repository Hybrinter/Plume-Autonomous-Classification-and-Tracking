# flight.payload.gimbal.outer

**Source:** `packages/flight/src/flight/payload/gimbal/outer.py`
**Kind:** pure module

## Purpose

The outer law forms the rate reference `r` from co-rotating feedforward, residual
rate, and proportional elevation error. TRACKING (live) and REWIND clip to the live
smear cap and the hardware slew.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `smear_cap_rad_s` | function | Live smear-limited `|r|` cap |
| `clip_rate` | function | Symmetric rate clip |
| `outer_rate` | function | Mode-dependent rate reference |

## Inputs and outputs

`outer_rate` takes predictor and residual rates, `e_hat`, `K_p`, arbiter mode, a
live-vision flag, elevation, science limb, hardware slew, live exposure, smear
budget, and band IFOV. It returns `r` in rad/s.

## Behavior

1. Compute the smear cap from live `exposure_us` and the smear pixel budget.
2. In REWIND, drive toward the science limb at the smear/hardware cap.
3. In TRACKING with a live filter, form `omega_t_nom + omega_t_res + K_p * e_hat`
   and clip.
4. Otherwise return `0.0` (cold TRACKING or unused SAFE path).

## Errors and faults

None.

## Messages

None.

## Configuration

`OuterLoopConfig.Kp` and `PreprocessingConfig.max_motion_smear_px` set the
proportional gain and smear budget. Hardware slew comes from `GimbalConfig`.

## Constraints

SAFE / STOW / HOME use the position loop and do not call this smear cap. The
function is pure.

## Related documents

- [`flight.payload.gimbal.predictor`](predictor.md)
- [`flight.payload.tracking.residual`](../tracking/residual.md)
- [`flight.payload.control`](../control.md)
