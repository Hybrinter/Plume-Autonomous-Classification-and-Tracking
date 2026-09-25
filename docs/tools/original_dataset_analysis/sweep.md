# tools.original_dataset_analysis.sweep

**Source:** `packages/tools/src/tools/original_dataset_analysis/sweep.py`
**Kind:** module

## Purpose

This module trains the native matrix, scores the best validation weights, and
writes the review bundle.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `CellResult` | class | Metrics and loss history for one cell |
| `run_native` | function | Train the given native cells |
| `write_mask_previews` | function | RGB tiles with a smoke overlay |
| `write_review` | function | Tables, metric charts, and loss curves |
| `train_native` | function | Command line for the native sweep |

## Inputs and outputs

`run_native(index, cache, order, cells, config, device) -> tuple[CellResult, ...]`.

`write_review(order, results, path) -> None`.

`train_native(argv) -> int`.

## Behavior

1. Splits follow location ids with seed 0 and fractions 0.70, 0.15, and 0.15.
2. Train moments are fit on the selected channels of the train split.
3. The classifier uses every tile. The segmentor drops tiles with no annotation
   file. The batch size is 64 and the loader uses no worker processes.
4. The headline score is PR-AUC for classification and Dice for segmentation.
   The JSON file keeps the other test metrics and the per-epoch losses.
5. ``write_review`` fills the native score column and leaves the ground-sample
   column blank. One bar chart is written per metric. One log-log loss chart
   is written per cell, with a line on the selected epoch.

## Errors and faults

`ValueError` when CUDA is requested and unavailable, a cell is not side 120,
a named cell is not in the native matrix, or a loss is missing.

## Messages

None.

## Configuration

``train-native`` accepts ``--images``, ``--labels``, ``--cache``, ``--out``,
``--device``, ``--only``, ``--preview``, and ``--epochs``. The default device
is ``cuda``. The default epoch count is 40.

## Constraints

The sweep trains native cells only. Ground-sample cells stay in the table
with a blank score.

## Related documents

- [`tools.original_dataset_analysis`](../original_dataset_analysis.md)
- [`tools.original_dataset_analysis.train`](train.md)
- [`tools.original_dataset_analysis.results`](results.md)
