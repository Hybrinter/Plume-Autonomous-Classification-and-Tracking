# tools.ml_models.analysis.summaries

**Source:** `packages/tools/src/tools/ml_models/analysis/summaries.py`
**Kind:** module

## Purpose

This module declares the two tagged, versioned summary records produced
by analysis executions: `DatasetSummary` and `ModelTrainingSummary`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SUMMARY_SCHEMA_VERSION` | constant | Supported schema version (1) |
| `SummaryStatus` | type alias | `COMPLETE`, `PARTIAL`, or `FAILED` |
| `DatasetSummary` | dataclass | Tagged evidence of one dataset analysis |
| `ModelTrainingSummary` | dataclass | Tagged evidence of one model/training analysis |
| `Summary` | type alias | Union of the two summary records |

## Inputs and outputs

Constructor arguments only; both records validate at construction and
raise `ValueError` on violation.

## Behavior

`DatasetSummary` carries `measurement_id`, a `DatasetIdentity`, a
`CodeIdentity`, a `config_digest`, metric/split/artifact/output tuples,
warnings, a status, and the `DATASET_ANALYSIS` tag at schema version 1.
`ModelTrainingSummary` additionally requires training and evaluation
`DatasetIdentity` records, a `CheckpointIdentity`, and an optional
`history` artifact reference, tagged `MODEL_TRAINING_ANALYSIS`. Split
records must agree with the summary's dataset and checkpoint hashes: a
conflict is a construction error, not a normalization. A `COMPLETE`
summary cannot carry a required output that is `UNAVAILABLE` or
`SKIPPED`; optional outputs may be missing with an explicit reason.
Metric names, output names, artifact paths, and split identities are
unique within a summary.

## Errors and faults

All violations raise `ValueError` at construction.

## Messages

None.

## Configuration

None.

## Constraints

- `measurement_id` and `config_digest` are strict lowercase SHA-256
  strings.
- `schema_version` other than 1 and a mismatched `summary_kind` are
  rejected.
- A dataset-analysis split cannot carry a checkpoint hash; a model
  summary's checkpoint must name the training dataset, and its splits
  must name the evaluation dataset and checkpoint.
- Assembly of these records from evidence lands with the analysis
  orchestration phase; this module is the schema.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.artifacts`](artifacts.md)
- [`tools.ml_models.analysis.dataset`](dataset.md)
- [`tools.ml_models.analysis.model`](model.md)
