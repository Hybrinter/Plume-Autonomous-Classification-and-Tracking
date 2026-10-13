# tools.ml_models.analysis.model_summary

**Source:** `packages/tools/src/tools/ml_models/analysis/model_summary.py`
**Kind:** module

## Purpose

This module builds the one canonical model/training summary bound to
frozen scientific evidence and source snapshots. The measurement
identity covers the checkpoint, both dataset identities, the resolved
scientific settings, code provenance, and every frozen numerical
record; rendering artifacts and style are excluded. A render-only
publisher retains the original measurement identity, values, and code
identity and never calls this builder.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `model_summary` | function | Assemble the canonical `ModelTrainingSummary` |

## Inputs and outputs

`model_summary(measured: ModelMeasurement, cfg: ModelAnalysisConfig,
code: CodeIdentity, artifacts: tuple[ArtifactRef, ...],
rendering_outputs: tuple[AvailabilityRecord, ...]) ->
Result[ModelTrainingSummary, str]` returns the validated summary on
`Ok`.

## Behavior

1. The frozen measurement's `config_digest` must equal
   `config_digest(cfg)`; a settings drift fails the binding.
2. `measurement_id` is a SHA-256 over the canonical identity payload:
   summary kind, both dataset identities, the checkpoint identity, the
   scientific config digest, code identity, the frozen per-split
   evidence (metrics, curves, strata, and capture references), the
   baseline, the generalization and development records, the frozen
   training/task/generalization figure recipes, the resource and
   metric-reference records, and the source-snapshot path/hash/size
   entries. Availability outputs and warnings are not hashed.
3. `split` outputs carry the per-split evidence identities; every
   measured split is indexed exactly once.

## Errors and faults

Returns `Err` when the supplied config digest disagrees with the
frozen measurement or when the assembled summary fails its own
contract validation.

## Messages

None.

## Configuration

`ModelAnalysisConfig`; see
[`tools.ml_models.analysis.config`](config.md).

## Constraints

- No inference, fitting, or resampling happens here; the function only
  binds frozen records into the summary contract.
- Rendering artifacts and `PlotConfig` values never enter
  `measurement_id`, so restyled renders reproduce it exactly.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.summaries`](summaries.md)
- [`tools.ml_models.analysis.model_measurement`](model_measurement.md)
- [`tools.ml_models.analysis.model`](model.md)
