# tools.analysis.plots.thermal

**Source:** `packages/tools/src/tools/analysis/plots/thermal.py`
**Kind:** module

## Purpose

The thermal plot builder renders temperature versus the effective limit, limit override, sample
activity, and over-limit fault counts.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `build` | function | Build thermal figures from a wide DataFrame |

## Behavior

1. Temperature versus effective limit overlay (`thermal.temperature_c` vs `thermal.limit_c`).
2. Commanded limit override line panel.
3. Stacked thermal samples and over-limit faults per step.
4. Cumulative thermal fault count.

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
