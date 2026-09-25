# tools.original_dataset_analysis.cli

**Source:** `packages/tools/src/tools/original_dataset_analysis/cli.py`
**Kind:** module

## Purpose

This module lists the native matrix, checks a JSON result table, or starts the
native training sweep.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `main` | function | Argument entry point |

## Inputs and outputs

`main(argv) -> int`.

``--descriptions`` is one string per GeoTIFF band. ``--results`` is an optional
JSON file of ``{task, subset, side_px}`` objects.

## Behavior

1. Descriptions are verified into a band order.
2. Without ``--results``, each planned cell is printed as task, subset, and side.
   ``--axis gsd`` lists the 12-band set and RGB at every legal side.
3. With ``--results``, the file must contain every planned cell for the chosen axis.
4. ``train-native`` passes the remaining arguments to the sweep.

## Errors and faults

`ValueError` when a description has no Sentinel-2 id or the result file omits
a cell. `json.JSONDecodeError` on malformed JSON.

## Messages

None.

## Configuration

None.

## Constraints

Listing and checking do not train a network. ``train-native`` does.

## Related documents

- [`tools.original_dataset_analysis`](../original_dataset_analysis.md)
- [`tools.original_dataset_analysis.matrix`](matrix.md)
- [`tools.original_dataset_analysis.sweep`](sweep.md)
