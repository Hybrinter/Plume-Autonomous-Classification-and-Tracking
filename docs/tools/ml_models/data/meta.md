# tools.ml_models.data.meta

**Source:** `packages/tools/src/tools/ml_models/data/meta.py`
**Kind:** module

## Purpose

This module stores pack identity and provenance, and hashes the processed-pack
files.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `NormName` | type | `normalize_dn`, `band_z`, or `unit` |
| `IngestPath` | type | Prism proxy, flight camera, or Sentinel-2 study |
| `Radiometry` | type | `s2_l2a_reflectance` or `normalize_dn` |
| `DatasetMeta` | class | Every field written to `dataset.json` |
| `Provenance` | class | Every field written to `provenance.json` |
| `provenance_from_meta` | function | Copy provenance fields off a dataset sidecar |
| `dataset_meta_from_provenance` | function | Build a dataset sidecar from provenance plus geometry |
| `compute_dataset_hash` | function | SHA-256 over pack file digests |
| `write_dataset_meta` / `load_dataset_meta` | function | `dataset.json` codec |
| `write_provenance` / `load_provenance` | function | `provenance.json` codec |

## Inputs and outputs

`DatasetMeta` fields are `dataset_hash`, `source_doi`, `n`, `height`, `width`,
`in_channels`, `band_names`, `norm`, `bit_depth`, `ingest_path`, `radiometry`,
`gsd_m`, `extent_m`, `weight_table_id`, `band_mean`, and `band_std`.

`Provenance` fields are `ingest_path`, `radiometry`, `gsd_m`, `extent_m`,
`weight_table_id`, `band_names`, `norm`, `bit_depth`, `band_mean`, and
`band_std`.

`compute_dataset_hash(pack_dir) -> str` returns lowercase hex.

`load_dataset_meta(path, *, pack_dir=None, verify=True) -> DatasetMeta`.

`load_provenance(path) -> Provenance`.

## Behavior

1. `write_dataset_meta` and `write_provenance` write every field. JSON objects
   use those field names. `band_names`, `band_mean`, and `band_std` are JSON
   arrays.
2. Load rejects a missing key and an unknown key.
3. `in_channels` equals `len(band_names)`. `n`, `height`, `width`,
   `in_channels`, and `bit_depth` are at least 1. `band_names` is non-empty.
4. `compute_dataset_hash` reads `images.npy`, `masks.npy`, `labels.npy`,
   `splits.json`, and `provenance.json`, in that sequence.
5. Each file is hashed in 8 MiB chunks. The pack hash is the SHA-256 of lines
   `name:file_sha256`. Each line ends with a newline.
6. `dataset.json` is not an input to the hash.
7. `load_dataset_meta` recomputes the hash when `pack_dir` is set and `verify`
   is true. A mismatch raises `ValueError`.
8. When `norm` is `band_z`, `band_mean` and `band_std` each have length
   `len(band_names)`. Each `band_std` value is greater than 0. These are the
   fitted moments. `BandStats(mean=band_mean, std=band_std)` rebuilds them for
   `apply_band_z` on a new sample.
9. When `norm` is `normalize_dn` or `unit`, `band_mean` and `band_std` are
   empty.

## Errors and faults

`ValueError` when sidecar keys do not match, a value has the wrong type, a
bound check fails, band moments do not match the recipe, or the recomputed
hash does not match `dataset_hash`.
`FileNotFoundError` when a hashed pack file is missing. `OSError` or
`json.JSONDecodeError` when a sidecar is missing or malformed.

## Messages

None.

## Configuration

`verify` defaults to true. The hash chunk size is 8 MiB. There is no TOML file.

## Constraints

The digest is lowercase hex. `norm`, `ingest_path`, and `radiometry` are closed
sets. `in_channels` equals the band-name count. `band_z` stores one mean and
one std per band. Other recipes store empty moment lists. The pack hash covers
`provenance.json`, including those moments.

## Related documents

- [`tools.ml_models.data`](../data.md)
- [`tools.ml_models.data.norm`](norm.md)
- [`tools.ml_models.data.pack`](pack.md)
- [`tools.ml_models.data.split`](split.md)
