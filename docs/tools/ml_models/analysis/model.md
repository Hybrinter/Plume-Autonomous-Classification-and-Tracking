# tools.ml_models.analysis.model

**Source:** `packages/tools/src/tools/ml_models/analysis/model.py`
**Kind:** module

## Purpose

This module is the model and training-analysis orchestration boundary.
It verifies every frozen source input, measures the selected checkpoint
once inside a private capture workspace, assembles the frozen evidence
bundle, renders all recipes, binds one canonical summary, and publishes
the bundle exclusively. A failure before publication leaves no output
behind; a failure during publication leaves the reserved directory with
an `.incomplete` marker so it is never mistaken for a finished bundle.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `analyze_model` | function | Measure, freeze, render, and publish one model-analysis bundle |

## Inputs and outputs

`analyze_model(cfg: ModelAnalysisConfig) -> Result[Path, str]` returns
the published bundle directory on `Ok`. The `analyze` CLI command calls
it.

## Behavior

1. `load_model_inputs` verifies the run, the selected checkpoint bytes,
   the training and evaluation datasets, and the recorded provenance,
   then rebuilds the model and objective before any inference.
2. `measure_model` runs once inside a temporary capture workspace; no
   user output directory exists yet.
3. `assemble_model_artifacts` freezes every codec document, table,
   capture file, and source snapshot into bundle bytes and references.
4. `render_model_outputs` renders the frozen training, task, and
   generalization recipes plus the combined prediction galleries with
   the configured `PlotConfig`.
5. `model_summary` binds the measurement identity, code identity, and
   every artifact reference into one canonical summary; `rendering.json`
   records the render identity for the applied `PlotConfig`.
6. `publish_bundle` reserves the output exclusively and writes the
   summary plus all bundle files. The output never overwrites an
   existing directory and never sits inside the run or either dataset
   root.

## Errors and faults

Returns `Err` when input verification, measurement, codec assembly,
rendering, summary binding, or publication fails. Publication errors
are the ones documented under
[`tools.ml_models.analysis.artifacts`](artifacts.md).

## Messages

None.

## Configuration

`ModelAnalysisConfig`; see
[`tools.ml_models.analysis.config`](config.md).

## Constraints

- All input verification happens before any model construction or
  inference; checkpoint bytes are hash-verified before deserialization.
- Nothing is written to the output path until every codec and renderer
  has succeeded; the capture workspace is a private temporary directory.
- Checkpoint bytes never enter the bundle; the recorded checkpoint hash
  is the identity.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.model_inputs`](model_inputs.md)
- [`tools.ml_models.analysis.model_measurement`](model_measurement.md)
- [`tools.ml_models.analysis.model_artifacts`](model_artifacts.md)
- [`tools.ml_models.analysis.model_render`](model_render.md)
- [`tools.ml_models.analysis.model_summary`](model_summary.md)
