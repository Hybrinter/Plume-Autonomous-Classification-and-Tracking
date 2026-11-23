# tools.ml_models.analysis.model_render

**Source:** `packages/tools/src/tools/ml_models/analysis/model_render.py`
**Kind:** module

## Purpose

This module renders frozen model-analysis recipes and republishes
model bundles without re-measuring. First publication renders the
persisted training, task, and generalization recipes and the combined
prediction capture with the configured `PlotConfig`. A render-only
pass decodes the same versioned documents from a verified bundle,
re-verifies every frozen identity against the summary, and renders
again without touching checkpoints, source datasets, measurements, or
any selection or bootstrap code. Only rendered artifact references and
`rendering.json` differ in the republished summary.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ModelRender` | dataclass | Rendered files, references, and availability records |
| `render_model_outputs` | function | Render all frozen recipes for one plot config |
| `rendering_document` | function | Canonical `rendering.json` bytes |
| `render_model_bundle` | function | The `Result` render-only boundary |

## Inputs and outputs

`render_model_outputs(training_figures, task_figures,
generalization_figures, previews, kind, cfg) -> Result[ModelRender,
str]` produces `figures/` and `visuals/` bytes with availability
records. `render_model_bundle(root: Path, summary:
ModelTrainingSummary, cfg: PlotConfig, out: Path) -> Result[Path,
str]` republishes a verified bundle to a fresh exclusive output.

## Behavior

1. `verify_bundle` runs before this boundary; every referenced file is
   read once and re-checksummed.
2. `model-evidence.json` must decode and reproduce the summary's
   checkpoint, dataset identities, and config digest exactly;
   `config.toml` must re-parse to a `ModelAnalysisConfig` with the
   same scientific digest.
3. The persisted `training-figure-data.json`,
   `task/<kind>/model-figure-data.json`,
   `generalization/model-figure-data.json`, and
   `prediction-manifest.json` documents decode to their versioned
   schemas; preview bytes must match the manifest hashes and captured
   row identities must match the summary splits.
4. Frozen recipes render through the existing training, model-family,
   and prediction renderers with the new `PlotConfig`; formats and
   style may change, scientific values cannot.
5. Rendered outputs must reproduce the frozen availability index
   exactly; `rendering.json` binds the measurement identity to a new
   render identity for the applied plot config.
6. The republished summary is the original with only `FIGURE`,
   `VISUAL`, and `rendering.json` references and rendered output
   records replaced; every other file is byte-identical.

## Errors and faults

Missing, corrupt, or identity-disagreeing frozen documents return `Err`
with an actionable fresh-analyze-required message. Render failures,
unsafe output locations, and publication conflicts return `Err`
without creating output. A failure during publication itself leaves
the reserved directory with an `.incomplete` marker so it is never
mistaken for a finished bundle.

## Messages

None.

## Configuration

`PlotConfig`; see [`tools.ml_models.analysis.config`](config.md).

## Constraints

- Rendering never calls inference, evaluation, scoring, selection,
  threshold fitting, matching, bootstrapping, or model loading.
- The output is exclusive and must not lie inside the evidence bundle.
- `measurement_id`, code identity, splits, metrics, and source
  snapshot bytes are preserved verbatim across re-renders.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.model_artifacts`](model_artifacts.md)
- [`tools.ml_models.analysis.plots.model`](plots/model.md)
- [`tools.ml_models.analysis.plots.common`](plots/common.md)
- [`tools.ml_models.analysis.visuals.predictions`](visuals/predictions.md)
