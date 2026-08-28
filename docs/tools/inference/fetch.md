# tools.inference.fetch

**Source:** `packages/tools/src/tools/inference/fetch.py`
**Kind:** module

## Purpose

This module pins Zenodo record 4250706, verifies local checksums, and optionally
downloads the smoke-plume corpus. It unpacks image and label tarballs and writes
a 4-band processed pack with frozen splits.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `DatasetFile` | class | One checksummed Zenodo file |
| `DatasetManifest` | class | Record metadata plus file list |
| `load_dataset_manifest` | function | Parse `data/manifests/zenodo_4250706.toml` |
| `file_md5` | function | md5 hex digest of a path |
| `verify_file` | function | Size and md5 check |
| `select_pact_bands` | function | Take B2/B3/B4/B8 from a 13-band stack |
| `to_model_domain` | function | Resize, scale, clip to `(4, H, W)` in [0, 1] |
| `download_file` | function | HTTP fetch with checksum |
| `extract_tarball` | function | Unpack a gzip tarball |
| `pair_sample_stems` | function | Match image and mask files by stem |
| `load_mask_plane` | function | Resize a mask to `(1, H, W)` in {0, 1} |
| `preprocess_planes` | function | Convert one stack |
| `preprocess_tree` | function | Pack image-only `.npy` or GeoTIFF files |
| `preprocess_pack` | function | Write a labeled pack with splits |
| `main` | function | CLI used by `scripts/fetch_smoke_plume_dataset.py` |

## Inputs and outputs

`load_dataset_manifest(path=None) -> DatasetManifest`.

`verify_file(path, expected_md5, expected_size) -> bool`.

`to_model_domain(planes, height, width, indices, dn_scale) -> np.ndarray`.

`pair_sample_stems(image_dir, label_dir) -> tuple[tuple[Path, Path], ...]`.

`preprocess_pack(...) -> int` sample count.

`main(argv=None) -> int`.

## Behavior

1. Parse the committed TOML manifest.
2. Print the dataset citation and DOI.
3. Report ok / missing / mismatch for each file under `data/raw/`.
4. Download only when `--download` is set.
5. `--preprocess` extracts `images.tar.gz` and `segmentation_labels.tar.gz`.
6. Pair files by stem, resize masks with nearest neighbor, and derive labels.
7. Write `images.npy`, `masks.npy`, `labels.npy`, `splits.json`, and
   `dataset.json` under `--processed-dir`.

## Errors and faults

`ValueError` on a checksum mismatch after download. `ImportError` when
`--preprocess` sees GeoTIFF files and rasterio is missing. `FileNotFoundError`
when preprocess finds no paired stacks.

## Messages

None.

## Configuration

Band indices default to `(1, 2, 3, 7)` (Sentinel-2 B2/B3/B4/B8). DN scale
defaults to 10000. Output size defaults to 256. Split fractions come from
`data/manifests/zenodo_4250706_splits.toml`.

## Constraints

Default CLI does not download. CI does not invoke `--download`. The corpus is
not stored in git. Tiny golden tensors live under
`packages/tools/tests/fixtures/`.

## Related documents

- [`tools.inference`](../inference.md)
- [`tools.inference.data`](data.md)
- [`tools.inference.split`](split.md)
- [`tools.inference.train`](train.md)
