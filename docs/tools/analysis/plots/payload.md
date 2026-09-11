# tools.analysis.plots.payload

**Source:** `packages/tools/src/tools/analysis/plots/payload.py`
**Kind:** module

## Purpose

The payload plot builder renders gimbal FSM, pointing, control rates, residual estimators, and
science output figures from the payload wide frame.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `build` | function | Build payload figures from a wide DataFrame |

## Inputs and outputs

None.

## Behavior

1. Gimbal arbiter FSM categorical timeline.
2. Pointing truth versus measured elevation.
3. Commanded versus measured elevation rate (`r` versus `y_m`).
4. Residual KF state (`e`, `omega_t_res`) and covariance diagonal plus trace.
5. Tracking and safety counters (miss count, blob count).
6. State flags (motion inhibited, stow switch, tracking).
7. Stacked inference, gimbal command, product, and fault output per step.

## Errors and faults

None.

## Messages

None.

## Configuration

None.

## Constraints

None.

## Related documents

- [`tools.analysis.plots`](plots.md)
- [`tools.analysis.plots.common`](common.md)
