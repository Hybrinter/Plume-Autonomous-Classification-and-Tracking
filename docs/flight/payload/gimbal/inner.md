# flight.payload.gimbal.inner

**Source:** `packages/flight/src/flight/payload/gimbal/inner.py`
**Kind:** pure module

## Purpose

The inner law turns a rate reference and encoder-rate estimate into motor torque.
It is a PI on rate error plus a computed-torque term `J_hat * v + B_hat * y_m`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `InnerResult` | dataclass | Clipped torque, integrator, acceleration, clip flag |
| `inner_step` | function | Advance the PI by one inner period |

## Inputs and outputs

`inner_step` takes `r`, `y_m`, integrator, `dt`, plant copies, gains, `tau_max`, and
a travel-stop flag. It returns `InnerResult`. Units are SI (rad/s, N·m).

## Behavior

1. Form rate error `r - y_m` and PI acceleration `v`.
2. Compute unsaturated torque and clip to `+-tau_max`.
3. Integrate the error only when torque is not clipped and the axis is not on a
   hardware stop.

## Errors and faults

None. Clip is a bound, not a fault.

## Messages

None.

## Configuration

Gains `kp` and `ki` come from `InnerLoopConfig`. Plant copies `J`, `B`, and
`tau_max` come from `GimbalConfig`.

## Constraints

The function is pure. Time arrives only as `dt_s`. `y_m` is the encoder-rate estimate
in the law.

## Related documents

- [`flight.payload.gimbal.rate_fit`](rate_fit.md)
- [`flight.payload.gimbal.position`](position.md)
- [`flight.payload.control`](../control.md)
