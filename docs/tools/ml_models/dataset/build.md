# tools.ml_models.dataset.build

**Source:** `packages/tools/src/tools/ml_models/dataset/build.py`
**Kind:** module

## Purpose

This module turns one raw tile source into a finished dataset: validated,
split, normalized, augmented shards plus a `dataset.json` manifest.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `build_dataset` | function | Build a dataset from any `RawSource` |
| `build_flight` | function | Build from a labeled flight tile directory |
| `build_zenodo` | function | Build from the Zenodo 4250706 archives |

## Inputs and outputs

`build_dataset(source, dest, spec) -> DatasetManifest`. `dest` must not
already exist.

`build_flight(source_dir, dest, spec=None) -> DatasetManifest`. The
default `BuildSpec` applies when `spec` is None. The flight
`gsd_reference_m` in `source.json` must equal `spec.gsd_reference_m`,
and no index row may carry `gsd_nominal` — nominal captures belong to a
custom research source built through `build_dataset`.

`build_zenodo(images_tar, labels_tar, weights_path, dest, spec=None,
bins=None) -> DatasetManifest`. `bins` selects GSD bins; `None` emits
`DEFAULT_BINS`. The weight table id is recorded on
`spec.weight_table_id`; a non-empty spec id that disagrees with the table
is rejected. Zenodo imports are lazy inside the wrapper.

## Behavior

1. `dest` must be absent. A sibling `.<name>.partial-*` temporary
   directory is created with `tempfile.mkdtemp`, filled, and renamed onto
   `dest` only after `dataset.json` is in place. A failure removes only
   that temporary directory; other partial directories are untouched.
2. The source `band_names` must equal `spec.input_bands` and `domain` must
   be `dn` or `unit`. The index must be non-empty with unique non-empty
   `tile_id` values and non-empty `group_id` values.
3. Per-row geometry: GSD components must be finite and positive, labels
   must be `0.0` or `1.0`, and `theta_g_deg` must be finite when present.
   Each ref's `theta_g_deg` and `gsd_nominal` copy onto the stored row.
   With no `extent_m` the tile size is the flight
   193 by 258. With `extent_m` `(lateral, along)`, the size is
   `round(extent / gsd)` per axis, and the stored GSD is `extent / pixels`.
4. `assign_group_splits` assigns one split per group. Val and test rows
   get only the `id` element. Train rows get `spec.augment.elements`
   intersected with `legal_elements(height, width)`; when the stored
   lateral and along-track GSD differ, the list is further reduced to the
   four axis-preserving elements, even on square tiles.
5. The `segmentor` task skips rows without a mask. One `ShardWriter` per
   `(task, split, H, W)` is preallocated.
6. `iter_tiles` is consumed in one pass. Each streamed `RawTileRef` must
   equal the indexed ref exactly. The image must match the expected
   `(bands, H, W)` shape and contain only finite pixels. A mask must be
   `(1, H, W)` and binary, present exactly when `has_mask`.
7. Each planned row applies its dihedral element to the unit image and
   mask, quantizes to uint16, and appends to its shard.
8. The manifest records the spec, bins, per-shard counts, GSD ranges, and
   the content hash, then `dataset.json` is written.

## Errors and faults

`FileExistsError` when `dest` exists before or during the build.
`FileNotFoundError` from `build_zenodo` when an archive or the weight
table is missing. `ValueError` on a band or domain mismatch, an empty
index, a duplicate or empty `tile_id`, an empty `group_id`, a non-finite
or non-positive GSD or extent, a non-binary label, a rounded size below 1,
an empty legal-element intersection, a stream that disagrees with the
index in order, shape, or mask, a non-finite image, a non-binary mask, a
`spec.weight_table_id` that disagrees with the loaded table, or no
selected rows. `OSError` when the destination cannot be created.

## Messages

None.

## Configuration

All build parameters come from `BuildSpec`. There is no TOML file in this
module.

## Constraints

The source is consumed in one forward pass; `index` and `iter_tiles` must
agree in order and content. Positive labels are `label >= 0.5`. This
module does not import torch.

## Related documents

- [`tools.ml_models.dataset`](../dataset.md)
- [`tools.ml_models.dataset.raw`](raw.md)
- [`tools.ml_models.dataset.spec`](spec.md)
- [`tools.ml_models.dataset.store`](store.md)
- [`tools.ml_models.dataset.manifest`](manifest.md)
- [`tools.ml_models.dataset.sources`](../dataset/sources.md)
- [`tools.ml_models.cli`](../cli.md)
