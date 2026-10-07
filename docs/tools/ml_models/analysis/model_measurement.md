# tools.ml_models.analysis.model_measurement

**Source:** `packages/tools/src/tools/ml_models/analysis/model_measurement.py`
**Kind:** module

## Purpose

This module is the selected-checkpoint measurement policy for the
evidence-first model workflow. The caller loads and verifies immutable
run, model, and dataset inputs first. Canonical train and validation
are development populations; test is measured once only under explicit
`final_test` after checkpoint and scoring settings are fixed. Original
training rows alone fit the baseline. Shared bounded capture budgets
cover every split, including external-baseline scalars. Every
reduction and selection recipe is frozen here for later pure
rendering.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ModelInputs` | dataclass | Verified model, manifests, identities, and source snapshots |
| `ModelCapture` | dataclass | Scalar split evidence plus bounded immutable capture bytes |
| `ModelResource` | dataclass | One resource result or explicit unavailability |
| `ModelMeasurement` | dataclass | All frozen scientific outputs for one analysis |
| `measure_model` | function | The `Result` measurement boundary |

## Inputs and outputs

`measure_model(inputs: ModelInputs, cfg: ModelAnalysisConfig,
workspace: Path) -> Result[ModelMeasurement, str]` evaluates each
configured split once through `evaluate_split` with a bounded
`CaptureSink`, fits the baseline on original training rows, derives
generalization, development, resource, and metric-reference evidence,
and freezes every training, task, and generalization figure recipe and
prediction gallery selection.

## Behavior

- Train and validation are always measured against the resolved
  datasets; when an external `cfg.dataset` override is supplied,
  cohorts it lacks surface as explicit unavailable states and no
  values are fabricated. Test is measured once, only when
  `cfg.final_test` is set, and is never used for checkpoint selection
  or baseline fitting.
- The baseline is fitted on original (unaugmented) training rows only.
- Each split's successful capture bytes are frozen into `ModelCapture`
  records under the caller-supplied temporary workspace; nothing is
  written to the user output path.
- Resources are measured per recorded input shape; unavailable states
  carry explicit reasons; no synthetic numbers are produced.
- The returned measurement holds no model or tensor handles.

## Errors and faults

Returns `Err` when split evaluation, capture closing, baseline
fitting, generalization measurement, or recipe freezing fails.

## Messages

None.

## Configuration

`ModelAnalysisConfig` (`score`, `capture`, `generalization`); see
[`tools.ml_models.analysis.config`](config.md).

## Constraints

- Measurement runs once per analysis; rendering must consume the
  frozen records only.
- Unavailable metric and figure states are preserved explicitly rather
  than dropped.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.evaluate`](evaluate.md)
- [`tools.ml_models.analysis.model_inputs`](model_inputs.md)
- [`tools.ml_models.analysis.model_artifacts`](model_artifacts.md)
- [`tools.ml_models.analysis.metrics.generalization`](metrics/generalization.md)
