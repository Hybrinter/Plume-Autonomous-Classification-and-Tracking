# tools.analysis.plots.electrical

**Source:** `packages/tools/src/tools/analysis/plots/electrical.py`
**Kind:** module

## Purpose

The electrical plot builder renders bus power versus the limit and over-limit fault activity.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `build` | function | Build electrical figures from a wide DataFrame |

## Inputs and outputs

None.

## Behavior

1. Bus power versus limit overlay (`electrical.power_w` vs `electrical.limit_w`).
2. Stacked electrical samples and over-limit faults per step.
3. Cumulative power over-limit fault count.

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
