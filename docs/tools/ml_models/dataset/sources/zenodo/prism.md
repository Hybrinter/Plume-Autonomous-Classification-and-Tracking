# tools.ml_models.dataset.sources.zenodo.prism

**Source:** `packages/tools/src/tools/ml_models/dataset/sources/zenodo/prism.py`
**Kind:** module

## Purpose

This module loads the AP-3200T prism weight table and mixes Sentinel-2 L2A
counts into BLUE, GREEN, and RED reflectance planes.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `WeightTable` | dataclass | Per-color band weights with a stable `id` |
| `load_weight_table` | function | Read a weight table TOML |
| `mix_prism` | function | `(C, H, W)` L2A counts to `(3, H, W)` reflectance |

## Inputs and outputs

`load_weight_table(path)` reads a TOML file with an `id` string and
`[blue]`, `[green]`, `[red]` tables of band-id weights. `mix_prism(
stack_dn, band_ids, table)` returns float32 `(3, H, W)`: each output plane
is the weighted sum of `clip(stack_dn / 10000, 0, 1)` over that color's
weights. A band absent from a color contributes 0.

## Behavior

1. The TOML root must hold exactly `id`, `blue`, `green`, and `red`.
2. Each color table is a non-empty map of band ids to finite weights that
   sum to 1 within `1e-6`. Loaded maps are immutable.
3. `mix_prism` rejects a stack whose channel count disagrees with
   `band_ids`, a repeated band id, and a table that names a band absent
   from `band_ids`.

## Errors and faults

`FileNotFoundError` on a missing table file. `ValueError` on a non-table
root, missing or extra keys, an empty `id`, an empty color table, a
non-numeric or non-finite weight, a sum that is not 1, or the `mix_prism`
contract violations above. `tomllib.TOMLDecodeError` on malformed TOML.

## Messages

None.

## Configuration

The committed weight table is curve-height readings of the AP-3200T solid
IR-cut figure, not a laboratory integral. There is no repo TOML; the path
arrives through the CLI `--weights-path` option.

## Constraints

The table `id` is recorded on the dataset manifest through
`spec.weight_table_id`. This module does not import torch or rasterio and
does not download archives.

## Related documents

- [`tools.ml_models.dataset.sources.zenodo`](../zenodo.md)
- [`tools.ml_models.dataset.sources.zenodo.bands`](bands.md)
- [`tools.ml_models.dataset.sources.zenodo.adapt`](adapt.md)
- [`tools.ml_models.dataset.build`](../build.md)
