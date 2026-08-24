# tools.analysis.plots.model_deploy

**Source:** `packages/tools/src/tools/analysis/plots/model_deploy.py`
**Kind:** module

## Purpose

The model_deploy plot builder renders deploy lifecycle state, active version, staged and
rollback flags, shape ranks, and transition activity.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `build` | function | Build model_deploy figures from a wide DataFrame |

## Inputs and outputs

None.

## Behavior

1. Deploy lifecycle state categorical timeline.
2. Active model version categorical timeline.
3. Staged, rollback retained, and factory-active flag line panel.
4. Staged input and output shape rank lines.
5. Stacked deploy state messages and model-corrupt faults per step.
6. Cumulative deploy state transitions.

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
