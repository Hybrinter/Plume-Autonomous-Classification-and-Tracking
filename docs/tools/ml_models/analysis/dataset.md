# tools.ml_models.analysis.dataset

**Source:** `packages/tools/src/tools/ml_models/analysis/dataset.py`
**Kind:** module
**Status:** stub

## Purpose

This module is the dataset-analysis orchestration boundary. It is
unavailable until the dataset evidence phase lands.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `analyze_dataset` | function | Dataset analysis boundary |

## Inputs and outputs

`analyze_dataset(cfg: DatasetAnalysisConfig) -> Result[Path, str]`.

## Behavior

The function currently returns `Err` with an explicit unavailable message
and creates no output directory. The `dataset analyze` CLI command calls
it.

## Errors and faults

Always `Err` while unimplemented.

## Messages

None.

## Configuration

`DatasetAnalysisConfig`; see
[`tools.ml_models.analysis.config`](config.md).

## Constraints

- No analysis bundle is written while the boundary is unavailable.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.cli`](../cli.md)
