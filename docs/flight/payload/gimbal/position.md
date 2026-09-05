# flight.payload.gimbal.position

**Source:** `packages/flight/src/flight/payload/gimbal/position.py`
**Kind:** pure module

## Purpose

The position loop writes a rate reference for STOW, HOME, and GOTO into the same
inner PI. The output saturates at `r_max` and is not smear-capped.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `position_rate` | function | Saturated `K_pos * (theta_cmd - theta_g)` |

## Inputs and outputs

Inputs are command elevation, current elevation, `K_pos`, and `r_max` in SI.
The return value is a rate reference in rad/s.

## Behavior

1. Compute `K_pos` times elevation error.
2. Clip to `+-r_max`.

## Errors and faults

None.

## Messages

None.

## Configuration

`ControllerConfig.K_pos` and `r_max_stow_deg_per_s` set the gain and saturation.

## Constraints

The function is pure. Pose motion still goes through the inner torque loop.

## Related documents

- [`flight.payload.gimbal.inner`](inner.md)
- [`flight.payload.gimbal.arbiter`](arbiter.md)
- [`flight.payload.control`](../control.md)
