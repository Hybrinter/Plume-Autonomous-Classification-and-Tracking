# tools.ml_models.dataset.spec

**Source:** `packages/tools/src/tools/ml_models/dataset/spec.py`
**Kind:** module

## Purpose

This module defines `BuildSpec`, the specification every dataset build
applies, and its TOML reader.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TASK_NAMES` | constant | `classifier` and `segmentor` |
| `BuildSpec` | class | Split, augment, tasks, bands, and GSD reference |
| `load_build_spec` | function | TOML reader |

## Inputs and outputs

`BuildSpec` is a frozen pydantic dataclass:

- `split`: `SplitRecipe` shared by every task and bin.
- `augment`: `AugmentRecipe` for train rows. The build intersects it with
  the elements legal for each tile.
- `tasks`: non-empty unique subset of `TASK_NAMES`. Defaults to both.
- `input_bands`: required channel names. Must equal the source band list.
  Defaults to `INPUT_BANDS`.
- `gsd_reference_m`: finite positive reference used by `to_model_gsd`.
  Defaults to `GSD_REFERENCE_M`.
- `weight_table_id`: class-weight table identifier. Empty when unused.

`load_build_spec(path) -> BuildSpec`. Recognized TOML keys are `split`,
`augment`, `tasks`, `input_bands`, `gsd_reference_m`, and
`weight_table_id`. Missing keys keep the dataclass defaults. The `split`
table takes `seed`, `train_fraction`, `val_fraction`, and
`test_fraction`. The `augment` table takes `elements`.

## Behavior

1. The TOML root must be a table; unknown keys are rejected at the root and
   inside the `split` and `augment` tables.
2. `tasks` must be non-empty, unique, and drawn from `TASK_NAMES`.
3. `input_bands` must be a non-empty list of strings.
4. The same spec applies to every source. Val and test rows are not
   augmented.

## Errors and faults

`OSError` / `tomllib.TOMLDecodeError` on a missing or malformed file.
`ValueError` when the root is not a table, a root or nested `split` /
`augment` key is unknown, or a field has the wrong shape. `ValueError` from `BuildSpec` when the task list is
empty, unknown, or duplicated, `input_bands` is empty, or
`gsd_reference_m` is not finite and positive.

## Messages

None.

## Configuration

The whole module is the `BuildSpec` TOML schema. Defaults match the
`SplitRecipe`, `AugmentRecipe`, `INPUT_BANDS`, and `GSD_REFERENCE_M`
defaults.

## Constraints

This module does not import torch. Field types reuse the split and augment
recipes.

## Related documents

- [`tools.ml_models.dataset`](../dataset.md)
- [`tools.ml_models.dataset.build`](build.md)
- [`tools.ml_models.dataset.split`](split.md)
- [`tools.ml_models.dataset.augment`](augment.md)
- [`tools.ml_models.cli`](../cli.md)
