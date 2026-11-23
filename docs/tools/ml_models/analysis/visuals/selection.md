# tools.ml_models.analysis.visuals.selection

**Source:** `packages/tools/src/tools/ml_models/analysis/visuals/selection.py`
**Kind:** module

## Purpose

This module selects deterministic bounded dataset galleries from frozen
measurement metadata. Whole gallery groups must fit the shared capture
budget; selections are order-invariant under a seeded priority and never
infer observation independence.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `DatasetGallery` | class | One selected gallery: identifier, family, variants, elements |
| `GalleryPlan` | class | Selected inputs, reserved bytes, and explicit family outputs |
| `select_dataset_galleries` | function | Bounded representative, augmentation, and GSD-pair selection |

## Inputs and outputs

`select_dataset_galleries(measured: DatasetMeasurement, cfg:
CaptureConfig) -> GalleryPlan` consumes only the frozen measurement and
returns the plan the capture stage executes verbatim.

## Behavior

- Seeded stratified round-robin selection ranks canonical variants by a
  SHA-256 priority over the seed and variant id, making results
  independent of input ordering.
- `representative` galleries pick bounded labeled inputs;
  `augmentation` galleries cover the stored dihedral elements legal for
  the tile shape; `same_observation_gsd` galleries pair variants that
  share a recorded observation id at differing GSDs. Pairs are never
  reduced to a misleading single-GSD view.
- A conservative byte reservation (raw float32 image plus uint8 mask
  allowance and an NPZ header margin) enforces
  `max_preview_images` and `max_capture_bytes` atomically per gallery.
- Every family reports an explicit plan output: `AVAILABLE` when a
  gallery was selected, `SKIPPED` when eligible inputs existed but the
  budget admitted none, and `UNAVAILABLE` when no eligible recorded
  inputs exist.

## Errors and faults

No `Err` paths: ineligible or over-budget galleries become explicit
plan outputs, not failures.

## Messages

None.

## Configuration

`CaptureConfig` (`max_preview_images`, `max_capture_bytes`,
`examples_per_family`, `seed`); see
[`tools.ml_models.analysis.config`](../config.md).

## Constraints

- All decisions use frozen metadata only; the selection never reads
  dataset files.
- A shared observation id is provenance; the plan makes no
  statistical-independence claim.

## Related documents

- [`tools.ml_models.analysis.visuals`](../visuals.md)
- [`tools.ml_models.analysis.dataset_previews`](../dataset_previews.md)
- [`tools.ml_models.analysis.visuals.dataset`](dataset.md)
