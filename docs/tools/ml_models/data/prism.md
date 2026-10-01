# tools.ml_models.data.prism

**Source:** `packages/tools/src/tools/ml_models/data/prism.py`
**Kind:** module

## Purpose

This module mixes Sentinel-2 L2A counts with the AP-3200T prism weights. It
writes a 76 px chip pack and a stored 193 by 258 tile pack. It does not
download archives.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `PROXY_SIDE_PX` | constant | 76 pixels |
| `PROXY_GSD_M` | constant | `1200 / 76` metres |
| `CHIP_HW` | constant | Chip `(76, 76)` |
| `TILE_HW` | constant | Tile `(193, 258)`, along-track then lateral |
| `FRAME_HW` | constant | Flight frame `(1544, 2064)` |
| `TILE_GRID` | constant | `8` by `8` tiles per flight frame |
| `GSD_MATCH_TOLERANCE` | constant | Relative ground-sample-distance gap `0.01` |
| `SOURCE_DOI` | constant | `10.5281/zenodo.4250706` |
| `WeightTable` | class | Identifier and per-color band weights |
| `load_weight_table` | function | Read the weight TOML |
| `mix_prism` | function | L2A counts to BLUE, GREEN, RED |
| `to_proxy_chip` | function | Native mix, resample, and mask at 76 |
| `write_prism_pack` | function | Annotated chips as one processed pack |
| `write_tile_pack` | function | One stored tile per source chip |
| `LocationSplit` | class | Union group ids and both packs' row indices |
| `union_location_split` | function | One location split for two packs |

## Inputs and outputs

`load_weight_table(path) -> WeightTable`.

`mix_prism(stack_dn, band_ids, table) -> np.ndarray` with shape `(3, H, W)`.

`to_proxy_chip(stack_dn, band_ids, table, polygons) -> tuple[np.ndarray, np.ndarray]`.
The image is `(3, 76, 76)`. The mask is `(1, 76, 76)`.

`write_prism_pack(images_tar, labels_tar, weights_path, dest, *, recipe=None) -> DatasetMeta`.

`write_tile_pack(source, dest, *, seed=0, recipe=None) -> DatasetMeta`. `source`
is a chip-pack directory or a `ProcessedPack`.

`union_location_split(left, right, *, recipe=None, gsd_tolerance=0.01) -> LocationSplit`.

## Behavior

1. `load_weight_table` reads `id`, `[blue]`, `[green]`, and `[red]`. Each
   color's weights sum to 1 within `1e-6`.
2. `mix_prism` computes `clip(stack_dn / 10000, 0, 1)`, then a weighted sum
   per color. Plane order is BLUE, GREEN, RED. A band absent from a color
   contributes 0.
3. `to_proxy_chip` mixes at the stack's native resolution, area-resamples the
   image to 76, and rasterizes percent polygons at 76.
4. `write_prism_pack` loads the weight table before it opens an archive.
   Each stack is fitted to 120 by 120 before `to_proxy_chip`. A short side
   is edge-padded. A long side is cropped from the origin. A row is a tile
   whose `polygons` value is not `None`. An empty tuple stays in the pack.
   A missing annotation is omitted. The label is 1 when the mask has a
   positive pixel, otherwise 0. Group ids are location ids.
5. The chip-pack provenance is ingest path `sentinel2_4250706_prism_proxy`,
   radiometry `s2_l2a_reflectance`, ground sample distance `1200 / 76`,
   extent 1200 m, the table id, band names `BLUE`, `GREEN`, `RED`, norm
   `unit`, and bit depth 12.
6. The module writes one polygon chip pack. It does not write a presence-only
   pack.
7. `CHIP_HW` is `(76, 76)`. `TILE_HW` is `(193, 258)`. `FRAME_HW` is
   `(1544, 2064)`. `TILE_GRID` is `(8, 8)`. `8 * 193 = 1544` along-track.
   `8 * 258 = 2064` lateral. A tile tensor is `(N, C, 193, 258)`.
8. `write_tile_pack` reads a 76 px chip pack. Each row becomes one tile.
   Label 0 builds a mosaic of same-split negative chips. Label 1 places that
   annotated chip on the mosaic at a random offset. The offset keeps one
   positive mask pixel inside the tile. Group ids are the source location
   ids, in source order.
9. The tile-pack provenance is ingest path `sentinel2_4250706_prism_tile`,
   radiometry `s2_l2a_reflectance`, ground sample distance `1200 / 76`,
   extent 1200 m, the source table id, band names `BLUE`, `GREEN`, `RED`,
   norm `unit`, and the source bit depth.
10. `union_location_split` requires equal band names, equal norm, and ground
    sample distance within 1 percent. The test is
    `abs(left - right) / max(left, right) <= gsd_tolerance`. It assigns one
    split to the union of the group ids. Left-pack ids come first. A group
    id may appear in both packs. The function does not call `concat_packs`.
    Spatial size is not compared.

## Errors and faults

`FileNotFoundError` when the weight table, an archive, or a pack file is
missing. `ValueError` when a color's weights do not sum to 1, when the table
names a band id missing from `band_ids`, when no tile is annotated, when
fewer than three location ids are present, or when a stack side is more than
2 pixels from 120. `ValueError` also when a tile source is not a 76 px
unit-reflectance BLUE/GREEN/RED pack, when a split has no negative chip, when
two packs disagree on bands, norm, or ground sample distance, or when a pack
has no group ids. `tomllib.TOMLDecodeError` on malformed TOML.

## Messages

None.

## Configuration

`data/manifests/ap3200t_s2_weights.toml` holds table id `ap3200t_s2_figure`.
Blue is B1 = 0.59 and B2 = 0.41. Green is B3 = 1. Red is B4 = 1. The file
records curve-height readings of the AP-3200T solid IR-cut figure. It is not
a laboratory integral. B5 and longer bands are absent. Their weight is 0.

`recipe` defaults to fractions 0.70 / 0.15 / 0.15 and seed 0.

## Constraints

The stored planes are unit-interval reflectance. CI does not need the Zenodo
tarball. The weight table must be present for a chip-pack write. A tile pack
is stored at 193 by 258. It is not an online 1544 by 2064 sample. Chip and
tile packs keep separate spatial sizes. `concat_packs` is not the union
split.

## Related documents

- [`tools.ml_models.data`](../data.md)
- [`tools.ml_models.data.grid`](grid.md)
- [`tools.ml_models.data.zenodo`](zenodo.md)
- [`tools.ml_models.data.pack`](pack.md)
- [`tools.ml_models.data.canvas`](canvas.md)
