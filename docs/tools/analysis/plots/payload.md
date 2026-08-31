# tools.analysis.plots.payload

**Source:** `packages/tools/src/tools/analysis/plots/payload.py`
**Kind:** module

## Purpose

The payload plot builder renders gimbal FSM, pointing, control rates, tracking estimators, and
science output figures from the payload wide frame.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `build` | function | Build payload figures from a wide DataFrame |

## Inputs and outputs

None.

## Behavior

1. Gimbal arbiter FSM categorical timeline.
2. Pointing truth versus measured azimuth and elevation.
3. Commanded versus driver gimbal rates.
4. Kalman state (errors and rates) and covariance diagonal plus trace.
5. EMA boresight error components and magnitude.
6. Tracking counters (miss count, vision-seen flag, blob count).
7. State flags (motion inhibited, stow switch, tracking, EMA initialized).
8. Stacked inference, gimbal command, product, and fault output per step.

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
