# tools.ml_models.dataset.sources.zenodo.fetch

**Source:** `packages/tools/src/tools/ml_models/dataset/sources/zenodo/fetch.py`
**Kind:** module

## Purpose

This module holds the pinned Zenodo 4250706 checksum manifest, file MD5
verification, and the download helper used by
`scripts/fetch_smoke_plume_dataset.py`. It downloads only on explicit
request.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `DatasetFile` | dataclass | One manifest file entry: key, size, md5, url |
| `DatasetManifest` | dataclass | Pinned record metadata and file checksums |
| `load_dataset_manifest` | function | Parse the TOML manifest |
| `file_md5` | function | Lowercase hex md5 of a file |
| `verify_file` | function | Size-and-md5 check for a local file |
| `download_file` | function | Verified HTTP download of one manifest entry |
| `main` | function | CLI printing citation and checksum status |

## Inputs and outputs

`load_dataset_manifest(path=None)` reads
`data/manifests/zenodo_4250706.toml` by default. `main(argv)` returns a
process exit code.

## Behavior

1. `main` prints the citation, DOI, and per-file `ok`, `missing`, or
   `mismatch` status under `--raw-dir`.
2. With `--download`, missing or mismatched files are fetched through
   `urllib.request` and verified again after download.

## Errors and faults

`verify_file` returns False for missing or mismatched files.
`download_file` raises `ValueError` on a post-download checksum mismatch
and `OSError` on network or write failures. Manifest parse failures
surface as `OSError`, `TOMLDecodeError`, or pydantic validation errors.

## Messages

Status lines and download notices print to stdout.

## Configuration

`--manifest` and `--raw-dir` override the committed defaults.

## Constraints

- Nothing downloads without `--download`.
- Pixel conversion and dataset construction live in the dataset package,
  not here.

## Related documents

- [`tools.ml_models.dataset.sources.zenodo`](../zenodo.md)
- [`tools.ml_models.dataset`](../../../dataset.md)
