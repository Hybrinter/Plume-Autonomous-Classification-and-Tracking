# tools.ml_models.analysis.dataset_figures

**Source:** `packages/tools/src/tools/ml_models/analysis/dataset_figures.py`
**Kind:** module

## Purpose

This module derives the complete supported inventory of dataset chart
recipes from a frozen `DatasetMeasurement`. It produces frozen
coordinate data — `DatasetFigure` and `FigureSeries` records — that a
later renderer can turn into images without ever re-reading the
dataset or re-fitting any measurement. The module performs no source
or model I/O and writes no files.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `FigureSeries` | class | Frozen x/y coordinates with support for one series |
| `DatasetFigure` | class | One standalone chart recipe with population and units |
| `dataset_figure_data` | function | Complete figure-coordinate inventory from frozen evidence |
| `FigureKind` | alias | `BAR`, `ECDF`, `HISTOGRAM`, or `MATRIX` |
| `Coordinate` | alias | `str | float | int` x-axis coordinate |

## Inputs and outputs

`dataset_figure_data(measured) -> tuple[DatasetFigure, ...]` takes the
frozen `DatasetMeasurement` produced by `measure_dataset` and returns
every supported figure recipe in a fixed order.

## Behavior

1. `dataset_counts` is a BAR figure over the named population metrics —
   stored rows, canonical task variants, canonical variants, recorded
   observations, total observations, and split groups. Null metric
   values (for example unrecorded total observations) stay `None`;
   they are never replaced by artificial zeros.
2. Coverage BAR figures pass through captured `CoverageCount` rows per
   split for `label`, `source_annotation_state`, `prepared_mask_state`,
   `mask_category`, `shape`, `bin_id`, `gsd_provenance`,
   `observation_id`, and `timestamp`. A recorded value is labeled with
   a `[recorded]` suffix; a null value gets its own bar — `Unbinned`
   for `bin_id`, `Unrecorded` otherwise — so an absent field is never
   conflated with a literal `"missing"` tag.
3. `class_prevalence` is a BAR of the positive-label fraction per
   split over canonical tile/GSD variants; an empty cohort yields a
   null y value.
4. ECDF figures cover lateral and along-track GSD, GSD anisotropy,
   local-GSD tile area, nonempty explicit-mask area (pixels, square
   metres, and fraction), and components per explicit mask, each with
   one series per split. Component-area ECDFs over all stored
   components use `COMPONENT` support. ECDFs are right-continuous with
   complete tie grouping: x values are the sorted distinct values, y
   the exact cumulative fraction, and support carries the full
   population count. A supported figure retains empty split series with
   zero support and no coordinates. There is no histogram estimation
   or sampling.
5. `mask_border_touching` is a BAR of the border-touching fraction per
   split over all explicit masks, including empty ones.
6. Per-band HISTOGRAM figures pass the captured `PixelMeasurement`
   histogram through verbatim: x carries the bin edges (one longer
   than the count vector) and y the exact counts, so restyling can
   never change the measurement. `pixel_moments` and `pixel_endpoints`
   are BARs of the captured band moments and endpoint counts.
   `pixel_correlations` is a MATRIX whose rows and columns follow
   `matrix_labels` exactly; constant-band pairs remain null.
7. Acquisition figures emit `month_utc` coverage for the
   `canonical_variant` and `recorded_observation` populations when any
   timestamp is recorded; an absent population yields a figure with an
   explicit `reason` and no series. `condition:*` coverage emits one
   BAR per recorded field and population; when no conditions exist a
   `conditions_unavailable` figure records that absence explicitly.

## Figure records

`FigureSeries` holds `name`, `x` coordinates, `y` values (nullable —
null means unavailable, not zero), and `support` (`MetricSupport` with
`IMAGE`, `COMPONENT`, or `PIXEL` units). `DatasetFigure` holds the
identifier, kind, title, axis labels, population description, series,
optional matrix and matrix labels/support, and an optional `reason`
declaring the figure unavailable; a renderer must surface and index
unavailable figures. Identifiers are stable filesystem-safe tokens
with a hash of the original label; display fields retain that label.

## Errors and faults

`dataset_figure_data` does not raise for missing evidence; absent
populations produce figures with an explicit `reason`.

## Messages

None.

## Configuration

None; all values come from the frozen `DatasetMeasurement`.

## Constraints

- No dataset traversal, tensor reads, model loading, or refitting —
  coordinates derive only from frozen evidence.
- The module emits coordinate records, not PNG/SVG files;
  `plots.dataset` renders the recipes, `dataset_previews` handles
  preview capture, and `dataset_render` orchestrates publication.
- Split cohorts partition canonical variants only, never stored
  augmentation rows; related GSD variants remain visible and are not
  treated as independent observations.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.dataset`](dataset.md)
- [`tools.ml_models.analysis.dataset_artifacts`](dataset_artifacts.md)
