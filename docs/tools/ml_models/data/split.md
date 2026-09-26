# tools.ml_models.data.split

**Source:** `packages/tools/src/tools/ml_models/data/split.py`
**Kind:** module

## Purpose

This module assigns each group id to train, val, or test, and stores the row
indices.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SplitRecipe` | class | Seed plus train/val/test fractions |
| `SplitIndex` | class | Integer row indices for each split |
| `assign_group_splits` | function | One split per group; rows follow the group |
| `write_splits` / `load_splits` | function | `splits.json` codec |

## Inputs and outputs

`SplitRecipe(seed=0, train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)`.

`assign_group_splits(group_ids, recipe) -> SplitIndex`.

`SplitIndex.train`, `.val`, and `.test` are `tuple[int, ...]`. Indices point
into the `group_ids` sequence.

`load_splits(path) -> SplitIndex`.

## Behavior

1. Fractions are finite, greater than 0, and sum to 1. Defaults are 0.70,
   0.15, and 0.15.
2. Unique group ids keep first-seen order. Fewer than 3 unique groups raises
   `ValueError`.
3. `numpy.random.default_rng(recipe.seed)` shuffles the groups. That object is
   a `numpy.random.Generator`.
4. Fractions apply to groups. Val and test each receive at least one group.
   When those two counts would consume every group, both counts become 1.
   Leftover groups go to train.
5. Every row whose group id is in a split stays in that split. Inside a split,
   row indices follow the input sequence.
6. `load_splits` rejects a missing name, a non-integer index, an unknown key,
   and overlapping indices.

## Errors and faults

`ValueError` when a fraction is not finite and positive, the fractions do not
sum to 1, fewer than 3 groups are present, or split indices overlap.

## Messages

None.

## Configuration

Default fractions are 0.70, 0.15, and 0.15. Default seed is 0. There is no
TOML file.

## Constraints

The split unit is the group id. Row indices inside a split follow the input
sequence. The shuffle uses numpy. This module does not import torch.

## Related documents

- [`tools.ml_models.data`](../data.md)
- [`tools.ml_models.data.pack`](pack.md)
- [`tools.ml_models.data.meta`](meta.md)
