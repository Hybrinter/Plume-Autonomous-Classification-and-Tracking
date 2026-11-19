# tools.ml_models.analysis.training

**Source:** `packages/tools/src/tools/ml_models/analysis/training.py`
**Kind:** module
**Status:** implemented

## Purpose

This module defines the strict incremental training records written by
the training loop, the sample-weighted epoch reduction, the validation
selection policy, and the inference-free history reader.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `LossObservation` | dataclass | Equal-image weighted objective term means |
| `observe_loss` | function | Reduce finite per-image loss terms to a `LossObservation` |
| `StepRecord` | dataclass | One attempted batch record |
| `EpochRecord` | dataclass | Sample-weighted record for one epoch |
| `reduce_epoch` | function | Combine one epoch's step records |
| `selected_metric` | function | Require an available validation score and its direction |
| `is_improvement` | function | Strict improvement; ties keep the earlier best |
| `CheckpointRecord` | dataclass | Serialized checkpoint identity plus bound validation |
| `EvaluationRecord` | dataclass | Train/val evidence at one serialized checkpoint |
| `TrainingExecution` | dataclass | Execution metadata and terminal status |
| `TrainingHistory` | dataclass | Parsed execution plus captured records and warnings |
| `read_training_history` | function | Parse a run's history without model or dataset access |

## Inputs and outputs

`observe_loss(components, pixel_weight, dice_weight) ->
Result[LossObservation, str]` reduces finite per-image tensors in
float64 without touching gradients.

`reduce_epoch(steps) -> Result[EpochRecord, str]` weights batch means by
actual sampled image counts, including AMP-skipped batches;
`optimizer_step` counts applied updates only.

`selected_metric(evidence, name) -> Result[(value, direction), str]`
requires validation-split evidence and a metric whose definition carries
a MINIMIZE or MAXIMIZE direction.

`read_training_history(run) -> Result[TrainingHistory, str]` parses
`execution.json` and `history.jsonl`.

## Behavior

- `LossObservation` requires nonnegative active components that sum to
  the total within tolerance.
- `StepRecord` carries global counters; throughput must agree with
  samples and duration, learning rates are nonnegative, and AMP scales
  are supplied together when present.
- `reduce_epoch` refuses mixed component activity inside one epoch and
  never averages unequal batch means equally.
- `CheckpointRecord` requires `checkpoints/best.pt` or
  `checkpoints/last.pt`, epoch and step on the identity, and a
  validation `SplitEvidence` whose task, dataset, and checkpoint hash
  agree and whose selected metric matches the recorded value and
  direction.
- `TrainingExecution` validates checkpoint consistency, the
  `selected_checkpoint` policy, and that COMPLETED executions carry
  best, last, and selected checkpoints plus a stop reason.
- `read_training_history` tolerates an unterminated final line only for
  non-COMPLETED executions. It verifies monotonic counters, contiguous
  batch and sample counts, epoch records that exactly match
  `reduce_epoch`, unique evaluated epochs, and that a COMPLETED
  execution ends on an evaluation whose checkpoint equals
  `execution.last`.

## Errors and faults

Record constructors raise `ValueError` on contract violations.
`observe_loss`, `reduce_epoch`, `selected_metric`, and
`read_training_history` return `Err` on invalid input or unreadable or
inconsistent history.

## Messages

None.

## Configuration

None.

## Constraints

- Records are pure evidence; no model or dataset access happens here.
- `torch` is imported inside `observe_loss` only.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.train.loop`](../train/loop.md)
