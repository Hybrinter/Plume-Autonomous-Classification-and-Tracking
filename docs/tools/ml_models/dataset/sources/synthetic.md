# tools.ml_models.dataset.sources.synthetic

**Source:** `packages/tools/src/tools/ml_models/dataset/sources/synthetic.py`
**Kind:** module

## Purpose

This module generates in-memory planted-blob tiles at the flight tile
size for dataset builds that need no corpus on disk.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SyntheticSource` | class | `RawSource` whose GSD values sample the science window |

## Inputs and outputs

`SyntheticSource(n=12, seed=0, label=None)`. `n` must be at least 3.
`seed` drives the background noise. `label`, when set, applies to every
tile; otherwise even indices are positive.

`index() -> tuple[RawTileRef, ...]` and `iter_tiles() -> Iterator[RawTile]`
return and yield the generated tiles in the same order.

Attributes: `name` `synthetic`, `band_names` `INPUT_BANDS`, `domain`
`dn`, `bit_depth` 12, empty `source_ref`, `extent_m` None, empty `bins`.

## Behavior

1. Each tile is uint16 noise on `(3, 193, 258)` with a bright block in the
   center quarter when the label is positive.
2. Every third tile carries a `(1, 193, 258)` mask, including annotated
   negatives.
3. `group_id` cycles `g0` through `g3` (fewer when `n` is below 4).
   `frame_id` equals `group_id`. `grid_rc` is `(index // 8, index % 8)`.
4. GSD cycles a four-entry window: `(15.87, 15.87)`, `(16.5, 17.1)`,
   `(18.6, 22.0)`, and `(23.3, 35.8)` metres.

## Errors and faults

`ValueError` when `n` is below 3 or `label` is not finite.

## Messages

None.

## Configuration

Constructor arguments only. There is no TOML file.

## Constraints

The source does not touch disk. This module does not import torch.

## Related documents

- [`tools.ml_models.dataset.sources`](../sources.md)
- [`tools.ml_models.dataset.raw`](../raw.md)
- [`tools.ml_models.dataset.geometry`](../geometry.md)
- [`tools.ml_models.dataset.build`](../build.md)
