# tools.ml_models.analysis.evaluate

**Source:** `packages/tools/src/tools/ml_models/analysis/evaluate.py`
**Kind:** module
**Status:** stub

## Purpose

This module is the boundary for one exhaustive split evaluation over
every row of a dataset split with one conditioned model into typed
`SplitEvidence`. It is unavailable until the evidence evaluation phase
lands.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `evaluate_split` | function | Exhaustive split evaluation boundary |

## Inputs and outputs

`evaluate_split(model, dataset, manifest, cfg, objective=None, capture=None)
-> Result[SplitEvidence, str]`.

## Behavior

The function currently returns `Err` with an explicit unavailable message.
No inference runs and no outputs are created.

## Errors and faults

Always `Err` while unimplemented.

## Messages

None.

## Configuration

`EvaluationConfig`; see [`tools.ml_models.analysis.config`](config.md).

## Constraints

- The boundary creates no output directories and runs no model calls.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.contracts`](contracts.md)
- [`tools.ml_models.analysis.capture`](capture.md)
