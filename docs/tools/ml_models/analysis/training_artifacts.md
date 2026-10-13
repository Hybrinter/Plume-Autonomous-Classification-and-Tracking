# tools.ml_models.analysis.training_artifacts

**Source:** `packages/tools/src/tools/ml_models/analysis/training_artifacts.py`
**Kind:** module

## Purpose

Frozen training-figure recipe serialization into bundle artifacts. The
codec turns caller-supplied `TrainingFigure` recipes into checksummed
`BundleFile` bytes: a canonical recipe document and a flat typed point
table in CSV and Parquet plus the table-schema manifest. It never
re-derives history coordinates, reads run histories, models, or
datasets, or renders anything; every value is copied from the frozen
records.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TRAINING_POINTS_SCHEMA` | constant | Column contract for `tables/training_points.{csv,parquet}` |
| `TABLE_SCHEMAS` | constant | Bundle path to schema mapping |
| `TrainingArtifacts` | dataclass | Returned file bytes and references |
| `training_artifacts` | function | `Result` codec boundary |

## Inputs and outputs

`training_artifacts(figures, *, prefix="") ->
Result[TrainingArtifacts, str]` takes frozen `TrainingFigure` recipes
and returns `files` (bundle bytes) plus `references` (checksummed
references). A nonempty `prefix` must be a safe relative POSIX path;
every returned file path and reference path is placed under
`prefix/...` while bytes, checksums, row counts, and the relative
paths inside the documents stay unchanged.

## Behavior

`training-figure-data.json` serializes the full recipe list under
`{"schema_version": 1, "figures": [...]}` — markers, scales, run
status, reasons, and warnings included. `tables/training_points.csv`
and `tables/training_points.parquet` carry one verbatim row per series
point with columns `figure`, `series`, `point_index`, `x`, `y`
(nullable), `n_exposures`, `exposure_unit`, `population`, and
`points_only`; raw zeros, nulls, and duplicate x coordinates are
copied unchanged. Repeated `n_exposures` values are per-series
metadata, not additive support. `tables/training_schema.json` declares
the shared column contract.

Table codecs stage in a private temporary directory; the returned
bytes are read back into `BundleFile` and bound to `ArtifactRef`
checksums. No summary or output reservation happens here.

## Errors and faults

Unsafe prefixes, malformed recipes that break the table schema, codec,
and serialization failures return `Err`; nothing is written outside
the private staging directory.

## Messages

None.

## Configuration

None.

## Constraints

- The writer copies frozen recipe values; it never re-derives
  coordinates, exposures, markers, or availability.
- The point table carries raw series values only; repeated exposure
  counts must not be summed across rows.
- `prefix` namespaces only the returned file and reference paths; it
  reserves no directories and changes no stored bytes.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.training_figures`](training_figures.md)
- [`tools.ml_models.analysis.artifacts`](artifacts.md)
