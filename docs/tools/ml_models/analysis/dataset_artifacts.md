# tools.ml_models.analysis.dataset_artifacts

**Source:** `packages/tools/src/tools/ml_models/analysis/dataset_artifacts.py`
**Kind:** module

## Purpose

This module freezes a caller-supplied `DatasetMeasurement` into an
exclusive, checksummed evidence bundle. It performs no dataset
traversal, reload, or recomputation: every artifact derives from the
already-frozen measurement record and the `DatasetAnalysisConfig`.
All codecs run inside a private temporary directory outside the
dataset, their bytes are read back into `BundleFile` objects, and a
single `publish_bundle` call prevalidates the references against those
bytes, exclusively reserves the destination, and writes the files
under an `.incomplete` marker.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SAMPLES_SCHEMA` | constant | `TableSchema` for `tables/samples.parquet` |
| `COMPONENTS_SCHEMA` | constant | `TableSchema` for `tables/components.parquet` |
| `COVERAGE_SCHEMA` | constant | `TableSchema` for `tables/coverage.csv` |
| `TABLE_SCHEMAS` | constant | Bundle path to `TableSchema` mapping |
| `code_identity` | function | Git revision and dirty state of the source worktree |
| `publish_dataset_measurement` | function | Measure-and-publish persistence boundary |

## Inputs and outputs

`publish_dataset_measurement(measured, cfg, *, extra_files=(),
extra_refs=(), extra_outputs=(), code=None) -> Result[Path, str]` takes
a frozen `DatasetMeasurement` and a `DatasetAnalysisConfig`, and on
success returns `Ok` with the published bundle directory.
`extra_files`/`extra_refs` carry pre-rendered artifacts bound by the
caller (figure bytes, preview captures, render metadata); they extend
the file and reference lists and are validated by `publish_bundle` like
every other artifact, so a reference without matching bytes or a
conflicting path is refused before reservation. `extra_outputs` merge
into the summary by name: duplicate names inside the batch are
rejected, a supplied name may replace only the base `figures` or
`visuals` record (for example flipping `figures` to `AVAILABLE`),
unique names such as `figure:<identifier>` records are appended, and
any attempt to replace a measured availability record such as
timestamps or conditions returns `Err`. `code` defaults to
`code_identity()` when not supplied.

`code_identity() -> CodeIdentity` runs `git rev-parse HEAD` and
`git status --porcelain --untracked-files=normal` with bounded timeouts
in the directory containing this module's file, not the caller's
working directory. When both commands succeed it records the revision
and whether the worktree is dirty; when unavailable it returns an
explicit `CodeIdentity(None, None, reason=...)`. `diff_hash` is never
claimed and no environment or credentials are read.

## Bundle layout

The published bundle contains the measurement-only layout below plus
any caller-supplied `extra_files`:

- `summary.json`: canonical tagged `DatasetSummary` encoded by
  `encode_summary` and written by `publish_bundle`.
- `config.toml`: the resolved `DatasetAnalysisConfig` written by
  `write_config`.
- `measurements.json`: canonical JSON (`sort_keys=True`,
  `allow_nan=False`, compact separators) of `asdict(measured)`. The
  full measurement structure is preserved verbatim: identity, the
  frozen `DatasetManifest` snapshot, samples (including `mask_key`),
  components, metrics, splits, coverage counts, pixel cohorts with
  complete per-band histograms and correlation matrices, baselines,
  duplicates, warnings, and the method version.
- `tables/samples.parquet`: one row per frozen `DatasetSample`.
- `tables/components.parquet`: one row per frozen `DatasetComponent`.
- `tables/coverage.csv`: one row per frozen `CoverageCount`.
- `tables/schema.json`: canonical JSON mapping each table path to
  `asdict` of its `TableSchema`, so render-only readers can reload the
  exact column contracts.

Every artifact is declared as a checksummed `ArtifactRef` inside the
summary; table refs carry exact row counts. The summary itself is
assembled by `dataset_summary`, which owns the measurement identity,
metric/split passthrough, and the base availability records — callers
that render figures or visuals pass `extra_outputs` to replace the base
`UNAVAILABLE` records.

## Table schemas

`SAMPLES_SCHEMA` columns, in order: `variant_id`, `dataset_hash`,
`task`, `split`, `height`, `width`, `row_index`, `tile_id`, `element`,
`group_id`, `bin_id`, `label` (`INT64`), `lateral_gsd_m`,
`along_gsd_m`, `gsd_anisotropy`, `tile_area_m2`, `gsd_nominal`
(`BOOLEAN`), `stored_rows`, `tasks_json` (JSON list), `image_sha256`,
`observation_id`, `acquired_at_utc`, `conditions_json` (JSON list of
condition tags), `annotation_source`, `annotation_version`,
`source_annotation_state`, `prepared_mask_state`, `mask_area_px`,
`mask_area_m2`, `mask_area_fraction`, `mask_components`,
`mask_border_touching`, `mask_task`, `mask_row_index`, `mask_element`.
The `task`/`split`/`row_index`/`element` columns describe the
representative stored key; the `mask_*` key columns name the segmentor
row that stores the measured mask and are null when no mask geometry
exists. Provenance and mask columns are nullable exactly as the frozen
record allows; absent metadata is null, never the literal tag
`"missing"`.

`COMPONENTS_SCHEMA` columns: `variant_id`, `component_index`,
`area_px` (`INT64`), `area_m2` (`FLOAT64`), copied verbatim from the
frozen component evidence.

`COVERAGE_SCHEMA` columns: `population`, `split`, `field`, `value`,
`n` (`INT64`), `total` (`INT64`), where `split` and `value` are
nullable. A null `value` means the metadata field was unrecorded; a
recorded literal `"missing"` remains a distinct value.

All scalar columns use the exact `INT64`/`FLOAT64`/`STRING`/`BOOLEAN`
dtypes declared in the schemas; row counts equal the frozen record
counts. Each table is stored in exactly one format — samples and
components as Parquet, coverage as CSV.

## Behavior

1. A private `tempfile.TemporaryDirectory` outside the dataset stages
   `config.toml`, `measurements.json`, the three tables, and
   `tables/schema.json`. Staged bytes are read into `BundleFile`
   objects; a codec or serialization failure returns `Err` and creates
   no output directory.
2. `dataset_summary` assembles the `DatasetSummary` from the frozen
   measurement, the config digest, the code identity, and the artifact
   refs.
3. `publish_bundle(out, summary, files, dataset_root=Path(cfg.dataset))`
   requires the file paths to match the references and validates each
   size and SHA-256 against the supplied bytes, exclusively reserves the
   destination, writes the files under an `.incomplete` marker, and
   removes the marker only when all writes succeed. Existing or raced
   destinations are refused without overwrite, output inside the source
   dataset is refused, and a mid-write failure leaves the `.incomplete`
   marker so `verify_bundle` refuses the partial bundle. `verify_bundle`
   is the separate post-publication check that reloads every artifact
   and recomputes checksums. User outputs are never cleaned up after a
   failure.

## Errors and faults

Returns `Err` when the config codec fails, canonical measurement JSON
cannot be encoded (for example non-finite values under
`allow_nan=False`), a table codec fails, the schema manifest cannot be
encoded, staging hits an `OSError`, `publish_bundle` refuses the
destination, or publication raises a path error such as an unwritable
or invalid output path. All failures are recoverable `Result` values;
the publisher never raises for user-correctable conditions.

## Messages

None.

## Configuration

`DatasetAnalysisConfig`; see
[`tools.ml_models.analysis.config`](config.md).

## Constraints

- The publisher never traverses or reloads dataset files; every byte
  comes from the frozen `DatasetMeasurement`, the config, and
  caller-supplied extra files.
- `DatasetMeasurement` is imported for type checking only; the
  `dataset_summary` import is local, avoiding a module cycle with
  `dataset.py`.
- `code_identity` runs at most once per call; a supplied `code` value
  is used verbatim.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.dataset`](dataset.md)
- [`tools.ml_models.analysis.artifacts`](artifacts.md)
