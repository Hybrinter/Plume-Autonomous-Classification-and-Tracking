# tools.analysis.plots.mechanical

**Source:** `packages/tools/src/tools/analysis/plots/mechanical.py`
**Kind:** module

## Purpose

The mechanical plot builder renders launch-lock state, engagement flag, and per-step lock and
gimbal motion activity.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `build` | function | Build mechanical figures from a wide DataFrame |

## Inputs and outputs

None.

## Behavior

1. Launch-lock state categorical timeline.
2. Launch-lock engaged line panel.
3. Stacked lock state messages, lock faults, and observed gimbal commands per step.
4. Cumulative launch-lock state publications.

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
