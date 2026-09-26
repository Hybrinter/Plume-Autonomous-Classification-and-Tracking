# tools.original_dataset_analysis.split

**Source:** `packages/tools/src/tools/original_dataset_analysis/split.py`
**Kind:** module

## Purpose

This module assigns every location id to one of train, validation, or test.
Public names are re-exported from
[`tools.ml_models.data.zenodo`](../ml_models/data/zenodo.md). The shuffle is
`assign_group_splits` in `tools.ml_models.data.split`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SplitRecipe` | class | Seed and site fractions |
| `LocationSplit` | class | Site assignment and stems |
| `assign_location_splits` | function | One split name per location |

## Inputs and outputs

`assign_location_splits(index, recipe) -> LocationSplit`.

## Behavior

1. Unique location ids are shuffled with ``recipe.seed``.
2. Fractions apply to sites. Each split receives at least one site when three
   or more sites exist.
3. Every tile of a location is listed under that location's split.

## Errors and faults

`ValueError` when a fraction is not positive, the fractions do not sum to 1,
or fewer than three locations are present.

## Messages

None.

## Configuration

Default fractions are 0.70, 0.15, and 0.15.

## Constraints

The split unit is the location id. Tiles are not shuffled independently.

## Related documents

- [`tools.original_dataset_analysis`](../original_dataset_analysis.md)
- [`tools.original_dataset_analysis.index`](index.md)
