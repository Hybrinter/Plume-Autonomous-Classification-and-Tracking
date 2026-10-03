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

`write_flight_tile_dir(dest, tiles, band_names=..., tile_hw=..., grid=...,
source_ref="", gsd_reference_m=...)` creates `dest` with `source.json`,
`index.jsonl`, `tiles/<tile_id>.npy`, and `masks/<tile_id>.npy`.

`source.json` keys are `schema`, `domain`, `image_dtype`, `band_names`,
`tile_hw`, `grid`, `source_ref`, and `gsd_reference_m`. `schema` is 2,
`domain` is `unit`, and `image_dtype` is `float32`. An `index.jsonl` row
carries `tile_id`, `frame_id`,
`row`, `col`, `group_id`, `label`, `has_mask`, `theta_g_deg`,
`gsd_lateral_m`, `gsd_along_m`, and the optional boolean `gsd_nominal`
(default False).

`FlightTileDir(root)` exposes `name` `flight`, `domain` `unit`, empty
`bins`, `band_names`, `tile_hw`, `grid`, `gsd_reference_m`, and
`source_ref` from `source.json`. `index()`
returns the refs in file order with the recorded `tile_hw`. `iter_tiles()`
yields `RawTile` rows in index order, reading one `.npy` at a time and
checking it against the recorded `(len(band_names), H, W)` shape.

## Behavior

1. The writer refuses an existing `dest`, empty `band_names`, a
   non-positive `gsd_reference_m`, a non-positive `tile_hw` or `grid`,
   and an empty tile list.
2. Header `band_names`, `tile_hw`, `grid`, and `gsd_reference_m` follow
   flight `InferenceConfig` by default. Tile images are float32 unit
   `(C, H, W)` in
   `band_names` order with finite pixels inside `[0, 1]`. Masks
   are uint8 `(H, W)` or `(1, H, W)` and stored as `(H, W)`.
3. A `tile_id` must be a file stem: non-empty, unique, not `.` or `..`,
   and free of path separators. `row` and `col` must be within the recorded
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
not a file stem, a grid index outside the recorded flight grid, a non-finite label or
`theta_g_deg`, non-positive GSD, a non-float32 image, an image pixel
outside `[0, 1]`, wrong image or mask dtype and shape, an
empty `group_id`, a malformed `source.json` or `index.jsonl`, a
`source.json` schema other than 2, a `domain` or `image_dtype` other than
`unit` and `float32`, a duplicate
`tile_id`, an empty index, or a missing or mistyped array file.
`OSError` / `json.JSONDecodeError` on a missing or malformed file.

## Messages

None.

## Configuration

Writer defaults: `band_names`, `tile_hw`, `grid`, and `gsd_reference_m`
come from flight `InferenceConfig`, and `source_ref` is empty. There is no
TOML file.

## Constraints

The imported finished-tile format follows flight geometry. This module does
not import torch.

## Related documents

- [`tools.ml_models.dataset.sources`](../sources.md)
- [`tools.ml_models.dataset.raw`](../raw.md)
- [`tools.ml_models.dataset.build`](../build.md)
