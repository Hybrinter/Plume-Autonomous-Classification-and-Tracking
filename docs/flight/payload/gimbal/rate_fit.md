# flight.payload.gimbal.rate_fit

**Source:** `packages/flight/src/flight/payload/gimbal/rate_fit.py`
**Kind:** pure module

## Purpose

`fit_rate` estimates angular rate at the newest encoder sample from a uniform ring.
The fit is a causal polynomial differentiator. It is not a two-point slope.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `fit_rate` | function | Returns `y_m` in rad/s at the newest sample |

## Inputs and outputs

`fit_rate(theta_rad, dt_s, n_omega=7, degree=2)` takes encoder elevations oldest to
newest and the inner period. It returns a float rate.

## Behavior

1. Return `0.0` when `dt_s` is not positive or the ring has fewer than two samples.
2. Use the last `n_omega` samples. Drop polynomial degree to `n-1` while the ring is
   short.
3. Place the newest sample at `tau = 0` and return the first derivative there.

## Errors and faults

None. A singular least-squares fit returns `0.0`.

## Messages

None.

## Configuration

`InnerLoopConfig.rate_fit_n` and `rate_fit_degree` select window length and degree.

## Constraints

The ring must be uniformly sampled at `T_in`. The estimator holds no filter state.

## Related documents

- [`flight.payload.gimbal.inner`](inner.md)
- [`flight.payload.control`](../control.md)
