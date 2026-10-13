# tools.ml_models.analysis.model_inputs

**Source:** `packages/tools/src/tools/ml_models/analysis/model_inputs.py`
**Kind:** module

## Purpose

This module verifies and loads every frozen input that
`measure_model` consumes. It fails closed on unsafe or inconsistent
sources before any model construction: source and output nesting,
existing outputs, linked source components, running executions, and
missing recorded checkpoints are all rejected.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `load_model_inputs` | function | `Result` input boundary for `analyze_model` |

## Inputs and outputs

`load_model_inputs(cfg: ModelAnalysisConfig) -> Result[ModelInputs,
str]` returns the typed `ModelInputs` record: the parsed training
history, the selected `CheckpointIdentity`, both dataset identities,
both manifests, the rebuilt model and objective, and the immutable
source byte snapshots.

## Behavior

1. `cfg.checkpoint` selects exactly one recorded `best` or `last`
   record; any other selector, a missing selected record, or an
   inconsistent recorded path is rejected. Only the requested record
   is required; a run whose other record is absent still loads. A
   still-running execution is rejected.
2. The run's flat `config.toml` is parsed strictly as `TrainConfig` and
   compared field-for-field with the recorded execution config.
3. The recorded training dataset path is checked against the config
   dataset. `dataset_identity` verifies dataset content and manifest
   hash; the manifest hash must equal the recorded validation
   `dataset_manifest_hash`, and a checkpoint without that hash is an
   actionable incompatibility.
4. Training provenance is recomputed through `training_provenance` and
   must equal the captured execution provenance under canonical JSON
   comparison.
5. The selected checkpoint file's actual bytes are SHA-256 hashed
   before deserialization, and the same verified bytes are loaded
   through `torch.load(weights_only=True)` on a `BytesIO` so no second
   disk read can race the check. The payload's kind, architecture,
   epoch, step, dataset hash, conditioning, metric, config, provenance,
   and embedded validation evidence must all agree with the recorded
   identity and execution; `samples_seen` must agree with the matching
   durable `EvaluationRecord` when one exists.
6. The architecture is resolved and built through the registry, the
   conditioning must match (`IgnoreGsd` selects the ignored id), and
   the state dict loads strictly. `build_loss` rebuilds the recorded
   objective.
7. Source snapshots carry `source/config.toml`, `source/execution.json`,
   `source/history.jsonl` when present, `source/training-dataset.json`,
   and `source/evaluation-dataset.json` byte-identically. Checkpoint
   bytes stay out of the bundle.

An external `cfg.dataset` override is verified the same way and must
share the training bands, order, normalization unit, image dtype, and
GSD reference while retaining its own identity.

## Errors and faults

Returns `Err` for missing or linked sources, unsafe selectors or
output locations, inconsistent recorded records, hash or provenance
mismatches, malformed checkpoint payloads, and strict state-load
failures. No output directory is created on any failure.

## Messages

None.

## Configuration

`ModelAnalysisConfig` (`run`, `out`, `checkpoint`, `dataset`); see
[`tools.ml_models.analysis.config`](config.md).

## Constraints

- Output paths must not already exist, must not be a dangling link, and
  must not lie inside the run or either dataset root.
- Symlink and junction components are rejected on the run, datasets,
  source files, and the checkpoint path, including ancestors.
- Torch imports stay lazy inside the boundary.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.model`](model.md)
- [`tools.ml_models.analysis.model_measurement`](model_measurement.md)
- [`tools.ml_models.analysis.training`](training.md)
