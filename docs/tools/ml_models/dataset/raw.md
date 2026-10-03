# tools.ml_models.dataset.raw

**Source:** `packages/tools/src/tools/ml_models/dataset/raw.py`
**Kind:** module

## Purpose

This module defines the raw tile contract a source must satisfy before a
dataset build reads it.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `Domain` | alias | `str`; the value `unit` |
| `GsdPair` | class | Pixel ground distance at a tile center, lateral then along-track; re-exported from `flight.payload.gimbal.footprint` |
| `BinSpec` | class | One named GSD bin recorded on the finished dataset |
| `RawTileRef` | class | Row identity and geometry without pixel arrays |
| `RawTile` | class | One raw tile: a `RawTileRef` plus image and optional mask |
| `RawSource` | protocol | Forward-only producer of raw tiles |

## Inputs and outputs

`RawSource.index() -> tuple[RawTileRef, ...]` returns one ref per tile.
`RawSource.iter_tiles() -> Iterator[RawTile]` yields each tile once, in
`index` order.

`RawSource` attributes: `name`, `band_names`, `domain`, `source_ref`, and
`bins`. `domain` is `unit`: a source supplies prepared float32 unit
pixels. `bins` is empty when the source has no named bins.

`RawTileRef` fields: `tile_id`, `group_id`, `label`, `has_mask`, `gsd`,
`height`, `width`, `frame_id`, `grid_rc`, `bin_id`, `theta_g_deg`, and
`gsd_nominal`. `theta_g_deg` is the gimbal elevation in degrees or None when the source
does not record one; `gsd_nominal` marks nominal orbit geometry when the
capture did not carry a measured value. `RawTile.image` is float32
`(C, H, W)` in the unit interval, where `(H, W)` is the indexed
`(height, width)`. `RawTile.mask` is `(1, H, W)` or None and is present
exactly when `has_mask` is True.

## Behavior

1. A source advertises its channel order and unit pixel domain, and
   indexes the actual image dimensions of every row.
2. `index` lists every row up front. `iter_tiles` makes one forward pass in
   the same order.
3. Labels at or above 0.5 count as positive. Rows that share a `group_id`
   stay in one split.
4. A source may emit more than one spatial size; `tile_hw` is a property
   of each row.

## Errors and faults

The protocol raises nothing itself. Implementations and the build raise
`ValueError` on contract violations.

## Messages

None.

## Configuration

None.

## Constraints

`iter_tiles` is forward-only and yields each tile once. Implementations
are structural `Protocol` matches; they do not inherit `RawSource`. This
module does not import torch.

## Related documents

- [`tools.ml_models.dataset`](../dataset.md)
- [`tools.ml_models.dataset.build`](build.md)
- [`tools.ml_models.dataset.sources`](../dataset/sources.md)
