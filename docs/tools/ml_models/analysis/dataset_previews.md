# tools.ml_models.analysis.dataset_previews

**Source:** `packages/tools/src/tools/ml_models/analysis/dataset_previews.py`
**Kind:** module

## Purpose

This module captures identity-bound compact dataset previews without
re-measurement. Selected inputs are copied from exact image and mask
keys, inverted to source orientation, and stored as checksum-bound NPZ
bytes. Only explicit stored masks are retained; a mask is never
synthesized for a variant that does not store one.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `DatasetPreview` | class | Captured input identity and display mapping for one NPZ file |
| `DatasetPreviewCapture` | class | Bounded immutable preview files plus the deterministic gallery plan |
| `display_channels` | function | Semantic display indices and label for band names |
| `capture_dataset_previews` | function | Bounded exact capture over the gallery plan |

## Inputs and outputs

`capture_dataset_previews(root: Path, measured: DatasetMeasurement,
cfg: CaptureConfig) -> Result[DatasetPreviewCapture, str]` reads the
dataset shards named by the gallery plan and returns the captured
previews, their `BundleFile` bytes, and the unchanged `GalleryPlan`.

`display_channels(band_names) -> tuple[tuple[int, ...], str]` returns
the display indices and the label for a band-name tuple. Names are
matched case-insensitively; when each of `BLUE`, `GREEN`, and `RED`
occurs exactly once among the recorded band names, the mapping uses
their recorded positions in RGB order and is labelled
`RGB (RED, GREEN, BLUE)`. Other three-or-more-channel inputs keep
recorded order and are labelled as not RGB; one- or two-band inputs
display their first band in grayscale.

## Behavior

- Each `DatasetPreview` carries the variant id, the bundle-relative
  `previews/` path, the SHA-256 and byte size of the exact NPZ payload,
  the frozen `DatasetSample`, and the display indices and label.
- Shared image and byte budgets are enforced by the plan; inputs reused
  by several galleries are captured once.
- Mask bytes are captured only when the stored segmentor row supplies
  explicit geometry; the measurement's `mask` state distinguishes an
  explicit empty mask from an absent one.

## Errors and faults

Returns `Err` when a selected row cannot be read or does not match its
frozen identity. Budget-excluded galleries are not errors: their plan
outputs record explicit `SKIPPED` states.

## Messages

None.

## Configuration

`CaptureConfig`; see
[`tools.ml_models.analysis.config`](config.md).

## Constraints

- Captured pixels are the exact canonical float32 unit arrays; no
  normalization, contrast stretch, resizing, or band synthesis is
  performed.
- Semantic display channel selection never alters the captured data.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.visuals.selection`](visuals/selection.md)
- [`tools.ml_models.analysis.visuals.dataset`](visuals/dataset.md)
- [`tools.ml_models.analysis.dataset_render`](dataset_render.md)
