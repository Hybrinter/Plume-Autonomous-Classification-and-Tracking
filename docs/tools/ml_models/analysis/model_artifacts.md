# tools.ml_models.analysis.model_artifacts

**Source:** `packages/tools/src/tools/ml_models/analysis/model_artifacts.py`
**Kind:** module

## Purpose

This module assembles the model evidence bundle: frozen documents,
typed tables, and capture bytes. Assembly is mechanical only. Existing
artifact codecs serialize the frozen training, task, and
generalization recipes and evidence; capture bytes are carried
verbatim; one combined prediction manifest covers every evaluated
split. `model-evidence.json` is the canonical frozen scientific
document that a render-only publisher re-verifies against the summary
without touching models, checkpoints, or source datasets.
`tables/split_metrics` persists every recorded cohort and stratum
`MetricValue` verbatim; compact scalar rows stay in the captured
`rows.jsonl` files.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SPLIT_METRICS_SCHEMA` | constant | `TableSchema` for flattened split metrics |
| `SPLIT_METRICS_TABLES` | constant | CSV and Parquet paths bound to the schema |
| `ModelArtifacts` | dataclass | Assembled files plus content-addressed references |
| `model_evidence_document` | function | Canonical `model-evidence.json` bytes |
| `combined_prediction_capture` | function | Union of all per-split preview captures |
| `assemble_model_artifacts` | function | The `Result` assembly boundary |

## Inputs and outputs

`assemble_model_artifacts(measured: ModelMeasurement, cfg:
ModelAnalysisConfig) -> Result[ModelArtifacts, str]` returns bundle
files and references on `Ok`; nothing is written to the caller's
output path.

## Behavior

1. `training_artifacts` serializes the frozen training recipes.
2. `model_figure_artifacts` serializes the task recipes under
   `task/<kind>/` and the generalization figure recipes under
   `generalization/`.
3. `generalization_artifacts` serializes each split's generalization
   evidence under `generalization/<split>/`; development evidence is
   bound to the validation record when present.
4. `prediction_artifacts` writes one combined
   `prediction-manifest.json` plus all normalized
   `capture/<split>/previews` NPZ bytes supplied by the captures.
5. Every capture file, including unused previews, is carried with its
   recorded references; snapshot files get `REFERENCE` kind records.
6. `model-evidence.json` (schema version 1) freezes the training and
   evaluation dataset identities, the checkpoint identity, the config
   digest, the resolved config, the baseline, generalization,
   development, resource, metric-reference, availability, and warning
   records, and the source snapshot hashes.
7. `tables/split_metrics.csv` and `tables/split_metrics.parquet`
   flatten each frozen `MetricValue` field per cohort and stratum row:
   split, task, population, stratum name and value, metric name, value,
   status, reason, unit, aggregation, threshold, support unit and `n`,
   canonical-JSON support counts, and optional frozen interval fields.
   `tables/split_metrics_schema.json` records the schema. No
   measurement or aggregation is performed.
8. Identical byte sets and identical references are deduplicated;
   conflicting entries are rejected. Every file has exactly one
   reference and every reference has exactly one file.

## Errors and faults

Returns `Err` when any codec, reference binding, table write, or
deduplication consistency check fails.

## Messages

None.

## Configuration

`ModelAnalysisConfig` (written verbatim to `config.toml`); see
[`tools.ml_models.analysis.config`](config.md).

## Constraints

- No scientific value is computed or re-derived here; the tables and
  documents only flatten or copy frozen records.
- Checkpoint bytes are never persisted; the checkpoint hash inside
  `model-evidence.json` is the identity.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.artifacts`](artifacts.md)
- [`tools.ml_models.analysis.model_measurement`](model_measurement.md)
- [`tools.ml_models.analysis.model_render`](model_render.md)
