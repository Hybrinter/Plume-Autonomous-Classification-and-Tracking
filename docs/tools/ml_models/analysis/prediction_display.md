# tools.ml_models.analysis.prediction_display

**Source:** `packages/tools/src/tools/ml_models/analysis/prediction_display.py`
**Kind:** module

## Purpose

Exact segmentation display arrays derived once from immutable cached
logits and explicit targets. The raw-mask threshold is copied from
frozen boundary provenance, not a render setting. The float64
sigmoid display does not claim float32 flight probability rounding
parity. Component IDs and matches remain frozen spatial records;
this module never matches or scores components again.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SegmentationDisplay` | dataclass | Read-only H-by-W truth/probability/predicted/FP/FN arrays plus the frozen threshold |
| `segmentation_display_data` | function | Freeze display transforms and verify cache/evidence agreement |

## Inputs and outputs

`segmentation_display_data(row, target, logits) ->
Result[SegmentationDisplay, str]` takes the captured segmentor row
and the exact float32 `(1, H, W)` target/logit arrays from the
verified preview cache.

## Behavior

Recomputes the display-only sigmoid probabilities and the raw binary
mask at the recorded `mask_probability_threshold`, derives
FP/FN/truth arrays, marks every array read-only, and rejects the
preview unless the recomputed target/predicted/TP/FP/TN/FN pixel
counts agree exactly with the frozen captured metrics.

## Errors and faults

Returns `Err` for non-segmentor rows or missing spatial evidence,
non-float32 or wrongly shaped arrays, nonfinite logits, non-binary
targets, an invalid recorded raw-mask threshold, or any disagreement
between cached pixels and frozen scalar counts.

## Messages

None.

## Configuration

None.

## Constraints

- Only threshold application and sigmoid scoring are applied;
  matching, component extraction, and aggregate metrics are never
  rerun.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.visuals.predictions`](visuals/predictions.md)
- [`tools.ml_models.analysis.metrics`](metrics.md)
