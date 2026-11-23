# tools.ml_models.analysis.model_figure_artifacts

**Source:** `packages/tools/src/tools/ml_models/analysis/model_figure_artifacts.py`
**Kind:** module

## Purpose

Frozen model-figure recipe serialization into bundle artifacts. The
codec turns caller-supplied `ModelFigure` recipes into checksummed
`BundleFile` bytes: a canonical recipe document, a flat typed point
table in CSV and Parquet, and the table-schema manifest. Every value
is copied from the frozen records; no sources, histories, models, or
datasets are read and no summary is published.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `MODEL_POINTS_SCHEMA` | constant | Column contract for `tables/model_points.{csv,parquet}` |
| `TABLE_SCHEMAS` | constant | Bundle path to schema mapping |
| `ModelFigureArtifacts` | dataclass | Returned file bytes and references |
| `model_figure_artifacts` | function | `Result` codec boundary |

## Inputs and outputs

`model_figure_artifacts(figures, *, prefix="") ->
Result[ModelFigureArtifacts, str]` takes frozen `ModelFigure` recipes
and returns `files` plus `references`. A nonempty `prefix` must be a
safe relative POSIX path; every returned path is placed under
`prefix/...` while bytes, checksums, and internal relative paths stay
unchanged.

## Behavior

`model-figure-data.json` serializes the full recipe list under
`{"schema_version": 1, "figures": [...]}` — identity, population,
operating points, matrices, ranges, reasons, and notes included.
`tables/model_points.csv` and `tables/model_points.parquet` carry one
verbatim row per series point with columns `figure`, `task`, `split`,
`dataset_hash`, `dataset_manifest_hash` (nullable), `checkpoint_hash`
(nullable), `series`, `point_index`, `x`, `y` (nullable), `lower`
(nullable), `upper` (nullable), `support_unit`, `support_n`,
`point_support_n` (nullable), and `style`. Matrices, notes, and
reasons live only in the JSON document; repeated `support_n` values
are per-series metadata, not additive support.
`tables/model_schema.json` declares the shared column contract.

Table codecs stage in a private temporary directory; failures return
`Err` before any caller output is touched.

## Errors and faults

Unsafe prefixes, malformed recipes that break the table schema, codec,
and serialization failures return `Err`.

## Messages

None.

## Configuration

None.

## Constraints

- The writer copies frozen recipe values only; it never re-derives
  coordinates, intervals, support, or availability.
- `prefix` namespaces only returned paths; it reserves no directories
  and changes no stored bytes.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.model_figures`](model_figures.md)
- [`tools.ml_models.analysis.artifacts`](artifacts.md)
