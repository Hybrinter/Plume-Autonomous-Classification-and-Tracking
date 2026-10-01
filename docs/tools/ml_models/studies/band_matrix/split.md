# tools.ml_models.studies.band_matrix.split

**Source:** `packages/tools/src/tools/ml_models/studies/band_matrix/split.py`
**Kind:** module

## Purpose

This module assigns every location id to one of train, validation, or test.

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

- [`tools.ml_models.studies.band_matrix`](../band_matrix.md)
- [`tools.ml_models.studies.band_matrix.index`](index.md)
