# tools.ml_models.analysis.training_figures

**Source:** `packages/tools/src/tools/ml_models/analysis/training_figures.py`
**Kind:** module

## Purpose

This module freezes a parsed `TrainingHistory` into standalone chart
recipes — `TrainingFigure` records — that a renderer can draw without
re-reading the run, re-averaging losses, re-selecting checkpoints, or
running inference. Attempted optimization batches and captured
sample-weighted epochs stay distinct from canonical train/validation
evaluations. Raw zeros, nulls, and repeated optimizer-step coordinates
stay frozen in the recipes. The module performs no source or model I/O
and writes no files.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `AxisScale` | alias | `linear` or `log` |
| `TrainingSeries` | dataclass | Raw event coordinates, exposure count, and population label |
| `TrainingMarker` | dataclass | Captured checkpoint/state x position with optional checkpoint hash |
| `TrainingFigure` | dataclass | One standalone recipe with scales, series, markers, run status, warnings, and optional unavailable reason |
| `training_figure_data` | function | Freeze optimization/evaluation/telemetry figures from validated history evidence |

## Inputs and outputs

`training_figure_data(history, *, final_test=None, checkpoint=None) ->
Result[tuple[TrainingFigure, ...], str]` takes a validated
`TrainingHistory`, an optional `SplitEvidence` for final test, and an
optional `CheckpointIdentity`; it returns the full recipe inventory in a
fixed order.

## Behavior

- Attempted-batch series use raw `StepRecord` coordinates: total loss
  (linear, semilog, and log-log variants), per-component losses, and
  telemetry (duration, throughput, gradient norm, update-applied
  flags, and AMP scaler before/after). Captured sample-weighted
  `EpochRecord` values produce separate epoch figures; no batch-mean
  re-averaging occurs.
- Canonical evaluation figures use `EvaluationRecord` train and
  validation evidence at their recorded optimizer steps: objective
  loss (three scale variants), weighted objective components, every
  shared captured metric, and a train-minus-validation development-gap
  figure per metric. Comparable metrics require one physical unit,
  aggregation, threshold, and support unit across captured states.
- An optional final-test evidence contributes exactly one
  `points_only` point at its verified selected checkpoint identity and
  step — never a per-epoch test line.
- Markers record the actual best and last evaluated checkpoint
  positions with their checkpoint hashes, plus the stopping state for
  non-running executions or the current running state at the last
  captured step.
- Figures with no captured values carry an explicit `reason`;
  `run_status` and history warnings travel on every recipe.
- Exposure counts use explicit `exposure_unit` labels (`IMAGE`,
  `BATCH`, `EPOCH`, `EVALUATION`, `PAIRED_EVALUATION`); they describe
  event populations, never independent observation support.

## Errors and faults

A final-test record that names an unrecorded checkpoint, mismatched
dataset/task identity, or an incompatible loss definition returns
`Err`, as do incompatible canonical metric definitions, changed
optimizer parameter-group layouts, and non-finite development gaps.

## Messages

None.

## Configuration

None; all values come from the frozen `TrainingHistory` and caller-
supplied evidence.

## Constraints

- No history re-parsing, loss re-reduction, interpolation, smoothing,
  resampling, or model inference — coordinates copy captured records.
- The module emits coordinate records, not images; `plots.training`
  renders the recipes and `training_artifacts` serializes them.
- Incomplete histories keep their captured prefix and run status;
  no completed execution is inferred.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.training`](training.md)
- [`tools.ml_models.analysis.plots.training`](plots/training.md)
- [`tools.ml_models.analysis.training_artifacts`](training_artifacts.md)
