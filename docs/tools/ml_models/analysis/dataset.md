# tools.ml_models.analysis.dataset

**Source:** `packages/tools/src/tools/ml_models/analysis/dataset.py`
**Kind:** module

## Purpose

This module measures one finished dataset and assembles its evidence
summary. Numerical helpers measure explicit masks and pixel
populations; `measure_dataset` freezes the whole dataset into a
`DatasetMeasurement` covering canonical tile/GSD variants, exact
coverage, baselines, and duplicate-content cohorts; `dataset_summary`
binds that frozen evidence to the config and code identity;
`analyze_dataset` is the measure-and-publish boundary that delegates
bundle persistence to `dataset_artifacts`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `measure_mask` | function | Geometry of one explicit binary mask |
| `MaskMeasurement` | class | Area, component, and border record for one mask |
| `measure_pixels` | function | Per-band moments, endpoint counts, histograms, and correlations |
| `BandMeasurement` | class | One band's population moments and counts |
| `PixelMeasurement` | class | Pixel-weighted cohort record with method and limitations |
| `DatasetSample` | class | One canonical tile/GSD variant unioned across task copies |
| `DatasetComponent` | class | One unfiltered four-connected mask component record |
| `CoverageCount` | class | Exact categorical support count |
| `PixelCohort` | class | Per-split or whole-population pixel evidence |
| `BaselineEvidence` | class | Train-derived prevalence or all-background baselines per split |
| `ContentDuplicate` | class | Identical image bytes across distinct canonical variants |
| `DatasetMeasurement` | class | Frozen whole-dataset evidence record |
| `measure_dataset` | function | Exact unsampled whole-dataset measurement |
| `dataset_summary` | function | Canonical summary assembly over frozen evidence |
| `analyze_dataset` | function | Measure-and-publish `Result` boundary |

## Inputs and outputs

`measure_mask(mask, gsd_m) -> Result[MaskMeasurement, str]` takes an
explicit uint8 binary `(1, H, W)` mask and a finite positive
`(lateral, along)` GSD in metres. `measure_pixels(images, band_names, *,
histogram_bins=32) -> Result[PixelMeasurement, str]` takes a non-empty
iterable of finite float32 unit `(C, H, W)` images — spatial sizes may
differ — and unique non-blank band names. `histogram_bins` is an `int`
of at least 2.

`measure_dataset(root) -> Result[DatasetMeasurement, str]` takes a
finished dataset directory whose `dataset.json` manifest and stored
`rows.jsonl` pass manifest/content-hash verification. It returns the
frozen `DatasetMeasurement` without writing anything.

`dataset_summary(measured, cfg, code, artifacts) -> DatasetSummary`
assembles the canonical summary from frozen values only; it never
re-reads dataset inputs.

`analyze_dataset(cfg: DatasetAnalysisConfig) -> Result[Path, str]` on
success returns the published evidence-bundle directory.

## Behavior

1. `measure_mask` labels unfiltered four-connected components of the
   foreground and returns the pixel area, area fraction, local-GSD area
   `foreground_pixels * lateral_GSD * along_GSD`, per-component pixel
   areas in label order, and whether the mask touches the border. An
   empty explicit mask measures zero area and zero components; a
   missing mask is an invalid input, never a manufactured negative.
   Components are connected pixel runs, not physical plumes.
2. `measure_pixels` weights every supplied pixel equally and merges
   centered image blocks in float64, yielding population mean and
   standard deviation per band, minima and maxima, counts of pixels
   exactly at 0 and 1, and exact equal-width `[0, 1]` histogram counts
   with left-closed bins and a right-closed last bin. Correlations are
   Pearson coefficients over the merged centered products, clamped to
   `[-1, 1]`; any pair involving a constant band is null, including the
   diagonal. The record carries a method string and limitations noting
   that pixels and source variants are correlated, not independent
   observations.
3. `measure_dataset` refuses symlinked or junctioned dataset members,
   verifies the manifest and content hashes, and validates every stored
   row. Each tile/bin/GSD/shape is measured once as a canonical variant
   unioned across task and augmentation copies: the representative
   `key` names the preferred stored view, while `mask_key` names the
   segmentor row that stores the explicit mask geometry. Images and
   masks are inverted to source orientation before measurement. Stored
   provenance (observation ids, acquisition timestamps, conditions,
   annotation source/version/state, prepared-mask state) is carried
   verbatim; missing metadata stays unknown. Coverage counts describe
   exact categorical support with null values distinct from the literal
   tag `"missing"`. Baselines score held-out splits with train-derived
   prevalence or all-background masks. Identical image bytes across
   variants form duplicate-content cohorts that are retained as
   evidence, not removed. A post-pass re-verifies the dataset identity
   and fails if the dataset changed during measurement.
4. `dataset_summary` hashes `asdict(measured)`, the config digest, and
   the code identity into the measurement id. Required measurement and
   table outputs are `AVAILABLE`; timestamps and conditions are
   `AVAILABLE` only when actually recorded in the dataset, otherwise
   explicitly `UNAVAILABLE` with a reason; figures and visuals are
   `UNAVAILABLE` until the rendering phase. Dirty or unknown code
   state adds a warning, never a fabricated identity.
5. `analyze_dataset` refuses an output inside the source dataset or an
   already-existing output, runs `measure_dataset`, then delegates to
   `publish_dataset_measurement`, which exclusively publishes one
   checksummed bundle; see
   [`tools.ml_models.analysis.dataset_artifacts`](dataset_artifacts.md).
   The `dataset analyze` CLI command remains unavailable until dataset
   rendering lands.

## Errors and faults

`measure_mask` returns `Err` for a non-array, wrong dtype or shape,
non-binary pixels, or non-positive/non-finite GSD, and for a ground
area outside the finite float range. `measure_pixels` returns `Err` for
an empty image tuple, empty or duplicate band names, `histogram_bins`
below 2, a non-float32 or non-`(C, H, W)` image, non-finite or
non-unit pixels, or a band-channel mismatch. `measure_dataset` returns
`Err` for a missing or tampered dataset, manifest or hash mismatch,
invalid stored rows, geometry-conflicting task copies, or a dataset
that changes during measurement. `analyze_dataset` returns `Err` for
an output inside the source dataset, an existing output, measurement
failure, or publication failure; it never overwrites user data.

## Messages

None.

## Configuration

`DatasetAnalysisConfig`; see
[`tools.ml_models.analysis.config`](config.md). `measure_pixels` takes
`histogram_bins` (default 32) as a keyword.

## Constraints

- Measurements are exact and unsampled: every canonical variant is
  measured once; nothing is deduplicated away or inferred from tile
  names.
- Missing observations, masks, and metadata stay explicit; the
  measurement never fabricates annotations or provenance.
- Observation totals depend only on recorded ids; a shared observation
  id is provenance, not a statistical-independence claim.
- Endpoint counts describe processed unit pixels, not raw sensor
  saturation.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.dataset_artifacts`](dataset_artifacts.md)
- [`tools.ml_models.dataset`](../dataset.md)
- [`tools.ml_models.cli`](../cli.md)
