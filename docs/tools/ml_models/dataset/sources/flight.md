# tools.ml_models.dataset.sources.flight

**Source:** `packages/tools/src/tools/ml_models/dataset/sources/flight.py`
**Kind:** module

## Purpose

This ground-side module reads and writes a labeled flight tile import
directory: a
`source.json` sidecar, an `index.jsonl` row list, and `tiles/` and
`masks/` arrays. The writer creates fixtures and compiled import examples; it
does not implement onboard frame collection or storage policy.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `FlightTileWrite` | class | One tile accepted by the writer |
| `write_flight_tile_dir` | function | Create a flight tile directory |
| `FlightTileDir` | class | `RawSource` reader over the directory |

## Inputs and outputs

`write_flight_tile_dir(dest, tiles, band_names=INPUT_BANDS, bit_depth=12,
source_ref="", gsd_reference_m=GSD_REFERENCE_M)` creates `dest` with `source.json`,
`index.jsonl`, `tiles/<tile_id>.npy`, and `masks/<tile_id>.npy`.

`source.json` keys are `band_names`, `bit_depth`, `source_ref`, and
`gsd_reference_m`. An `index.jsonl` row carries `tile_id`, `frame_id`,
`row`, `col`, `group_id`, `label`, `has_mask`, `theta_g_deg`,
`gsd_lateral_m`, `gsd_along_m`, and the optional boolean `gsd_nominal`
(default False).

`FlightTileDir(root)` exposes `name` `flight`, `domain` `dn`, `extent_m`
None, empty `bins`, and `gsd_reference_m` from `source.json`. `index()`
returns the refs in file order. `iter_tiles()` yields `RawTile` rows in
index order, reading one `.npy` at a time.

## Behavior

1. The writer refuses an existing `dest`, a `bit_depth` below 1, a
   non-positive `gsd_reference_m`, and an empty tile list.
2. Frame dimensions, grid limits, GSD reference, and default bands follow
   flight `InferenceConfig`. Tile images are uint16 `(3, 193, 258)` in
   `band_names` order. Masks
   are uint8 `(193, 258)` or `(1, 193, 258)` and stored as `(193, 258)`.
3. A `tile_id` must be a file stem: non-empty, unique, not `.` or `..`,
   and free of path separators. `row` and `col` must be within the configured
   flight grid.
4. A missing `group_id` in `index.jsonl` defaults to `frame_id`.
5. `grid_rc` is the `(row, col)` pair. `theta_g_deg` passes through to
   the ref, and `bin_id` is `elevation{nearest}` for the nearest of
   (5, 15, 25, 35, 45) degrees, ties choosing the smaller value.
6. `gsd_nominal` marks tiles whose GSD is nominal orbit geometry;
   `build_flight` rejects them, while a custom research source built
   through `build_dataset` may keep the flag.

## Errors and faults

`FileExistsError` when `dest` exists. `ValueError` on a `tile_id` that is
not a file stem, a grid index outside the configured flight grid, a non-finite label or
`theta_g_deg`, non-positive GSD, wrong image or mask dtype and shape, an
empty `group_id`, a malformed `source.json` or `index.jsonl`, a duplicate
`tile_id`, an empty index, or a missing or mistyped array file.
`OSError` / `json.JSONDecodeError` on a missing or malformed file.

## Messages

None.

## Configuration

Writer defaults: `band_names` and `gsd_reference_m` come from flight
`InferenceConfig`, `bit_depth` is 12, and `source_ref` is empty. There is no
TOML file.

## Constraints

The imported finished-tile format follows flight geometry. This module does
not import torch.

## Related documents

- [`tools.ml_models.dataset.sources`](../sources.md)
- [`tools.ml_models.dataset.raw`](../raw.md)
- [`tools.ml_models.dataset.geometry`](../geometry.md)
- [`tools.ml_models.dataset.build`](../build.md)
