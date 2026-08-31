# tools.inference.fetch

**Source:** `packages/tools/src/tools/inference/fetch.py`
**Kind:** module

## Purpose

This module pins Zenodo record 4250706, verifies local checksums, and optionally
downloads the smoke-plume corpus. `--preprocess` streams the image and label
archives into two 4-band processed packs with frozen splits.

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
| `preprocess_planes` | function | Convert one stack |
| `ZenodoIndex` | class | Image stems, positives, and annotated subset |
| `read_annotation_archive` | function | Polygon vertices keyed by stem |
| `index_image_archive` | function | Stem index over `images.tar.gz` |
| `preprocess_zenodo_archives` | function | Stream both archives into two packs |
| `main` | function | CLI used by `scripts/fetch_smoke_plume_dataset.py` |

## Inputs and outputs

`load_dataset_manifest(path=None) -> DatasetManifest`.

`verify_file(path, expected_md5, expected_size) -> bool`.

`to_model_domain(planes, height, width, indices, dn_scale) -> np.ndarray`.

`read_annotation_archive(labels_archive) -> dict[str, tuple[np.ndarray, ...]]`.

`preprocess_zenodo_archives(...) -> tuple[int, int]` classifier and segmentor
sample counts.

`main(argv=None) -> int`.

## Behavior

1. Parse the committed TOML manifest through the `DatasetManifest` schema.
2. Print the dataset citation and DOI.
3. Report ok / missing / mismatch for each file under `data/raw/`.
4. Download only when `--download` is set.
5. `--preprocess` streams `images.tar.gz` and `segmentation_labels.tar.gz`.
6. Rasterize polygon annotations, resize with nearest neighbor, and derive
   presence labels from class directories.
7. Write a classifier pack under `--classifier-dir` (default
   `<processed-dir>/classifier`) for every image.
8. Write a segmentor pack under `--segmentor-dir` (default
   `<processed-dir>/segmentor`) for the annotated subset.
9. Each pack holds `images.npy`, `masks.npy`, `labels.npy`, `splits.json`,
   `stems.json`, and `dataset.json`.

## Errors and faults

`ValueError` on a checksum mismatch after download, or when a requested pack
has no selected samples. `ImportError` when `--preprocess` needs rasterio and
it is missing. `FileNotFoundError` when an archive is missing.

## Messages

None.

## Configuration

Band indices default to `(1, 2, 3, 7)` (Sentinel-2 B2/B3/B4/B8). DN scale
defaults to 10000. Output size defaults to 256. Split fractions come from
`data/manifests/zenodo_4250706_splits.toml`. Presence labels cover all 21,350
images. Polygon masks cover 1,437 of them.

## Constraints

Default CLI does not download. CI does not invoke `--download`. The corpus is
not stored in git. Tiny golden tensors live under
`packages/tools/tests/fixtures/`. Preprocess reads archives as streams. It does
not extract them to disk.

## Related documents

- [`tools.inference`](../inference.md)
- [`tools.inference.annotations`](annotations.md)
- [`tools.inference.data`](data.md)
- [`tools.inference.split`](split.md)
- [`tools.inference.train`](train.md)
