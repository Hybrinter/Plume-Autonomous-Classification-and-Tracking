# tools.ml_models.data.prism

**Source:** `packages/tools/src/tools/ml_models/data/prism.py`
**Kind:** module

## Purpose

This module mixes Sentinel-2 L2A counts with the AP-3200T prism weights and
writes a 76 px polygon pack. It does not download archives.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `PROXY_SIDE_PX` | constant | 76 pixels |
| `PROXY_GSD_M` | constant | `1200 / 76` metres |
| `SOURCE_DOI` | constant | `10.5281/zenodo.4250706` |
| `WeightTable` | class | Identifier and per-color band weights |
| `load_weight_table` | function | Read the weight TOML |
| `mix_prism` | function | L2A counts to BLUE, GREEN, RED |
| `to_proxy_chip` | function | Native mix, resample, and mask at 76 |
| `write_prism_pack` | function | Annotated tiles as one processed pack |

## Inputs and outputs

`load_weight_table(path) -> WeightTable`.

`mix_prism(stack_dn, band_ids, table) -> np.ndarray` with shape `(3, H, W)`.

`to_proxy_chip(stack_dn, band_ids, table, polygons) -> tuple[np.ndarray, np.ndarray]`.
The image is `(3, 76, 76)`. The mask is `(1, 76, 76)`.

`write_prism_pack(images_tar, labels_tar, weights_path, dest, *, recipe=None) -> DatasetMeta`.

## Behavior

1. `load_weight_table` reads `id`, `[blue]`, `[green]`, and `[red]`. Each
   color's weights sum to 1 within `1e-6`.
2. `mix_prism` computes `clip(stack_dn / 10000, 0, 1)`, then a weighted sum
   per color. Plane order is BLUE, GREEN, RED. A band absent from a color
   contributes 0.
3. `to_proxy_chip` mixes at the stack's native resolution, area-resamples the
   image to 76, and rasterizes percent polygons at 76.
4. `write_prism_pack` loads the weight table before it opens an archive. A
   row is a tile whose `polygons` value is not `None`. An empty tuple stays
   in the pack. A missing annotation is omitted. The label is 1 when the
   mask has a positive pixel, otherwise 0. Group ids are location ids.
5. The pack provenance is ingest path `sentinel2_4250706_prism_proxy`,
   radiometry `s2_l2a_reflectance`, ground sample distance `1200 / 76`,
   extent 1200 m, the table id, band names `BLUE`, `GREEN`, `RED`, norm
   `unit`, and bit depth 12.
6. The module writes one polygon pack. It does not write a presence-only pack.

## Errors and faults

`FileNotFoundError` when the weight table or an archive is missing.
`ValueError` when a color's weights do not sum to 1, when the table names a
band id missing from `band_ids`, when no tile is annotated, or when fewer
than three location ids are present. `tomllib.TOMLDecodeError` on malformed
TOML.

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
tarball. The weight table must be present for a pack write.

## Related documents

- [`tools.ml_models.data`](../data.md)
- [`tools.ml_models.data.grid`](grid.md)
- [`tools.ml_models.data.zenodo`](zenodo.md)
- [`tools.ml_models.data.pack`](pack.md)
