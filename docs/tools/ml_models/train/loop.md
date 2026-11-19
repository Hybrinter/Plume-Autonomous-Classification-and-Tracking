# tools.ml_models.train.loop

**Source:** `packages/tools/src/tools/ml_models/train/loop.py`
**Kind:** module
**Status:** implemented

## Purpose

This module is the imperative training shell. `train` validates one
finished dataset, reserves an exclusive run directory, and drives a
seeded batch loop that persists durable evidence records, evaluated
checkpoints, and a terminal execution state.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `train` | function | Public `Result[Path, str]` boundary |

## Inputs and outputs

`train(cfg=None) -> Result[Path, str]`. `cfg=None` uses `TrainConfig`
defaults. `Ok` carries the run directory path; `Err` carries a message
naming the run when one was reserved.

## Behavior

1. A blank `dataset` fails before any output. `run_id` must be a single
   safe path component (no separators, colon, `.`, or `..`); an empty
   `run_id` is generated from `kind`, `config_digest`, and a timestamp.
2. The dataset manifest is loaded and hash-verified, measured training
   provenance traverses every shard's row metadata including test rows,
   and the task must have train and val shards — all before reservation.
   Test tensors are never loaded and no test evaluation runs.
3. Run and `checkpoint_path` destinations inside the dataset root are
   refused. An existing run directory is refused even with
   `overwrite=True`. An existing copy destination, including a symlink,
   is refused up front.
4. The run directory owns `checkpoints/`, `config.toml`,
   `execution.json`, `history.jsonl`, and `resources.json`. No
   `summary.json` is written.
5. `torch.manual_seed` receives `cfg.seed`. `make_loader` is seeded with
   `cfg.seed + epoch` and yields `ceil(train_samples / batch_size)`
   batches per epoch. Optimizer is SGD or AdamW; `cosine` adds a
   `CosineAnnealingLR` stepped once per epoch.
6. Each attempted batch records a `STEP` line: pre-update learning
   rates, the configured `per_sample_components` objective observation,
   counters, duration, throughput, `update_applied`, and optional
   gradient norm and AMP scales. Records are flushed with `fsync`
   immediately, and `execution.json` is atomically rewritten each step,
   epoch, and evaluation.
7. Targets must be exact binary values, output shape must match, and
   outputs and per-image losses must be finite before `backward`.
   Without AMP a non-finite gradient fails the run. With AMP on CUDA a
   scaler-skipped update is allowed: `optimizer_step` counts applied
   updates only, `update_applied` is false, and a warning is recorded.
   `samples_seen` grows by the real batch size either way. Step and
   epoch durations and throughputs cover only the optimization batch
   work between the CUDA sync fences; they exclude loader sampling,
   evidence `fsync`, and the evaluation and resource passes.
8. AMP autocast applies only when `cfg.amp` and a CUDA device are both
   present; AMP requested on CPU records a warning and reports
   `amp_enabled=False`. Gradient L2 norms are recorded only when
   `gradient_diagnostics` is set and the update was applied; AMP scales
   only when `amp_diagnostics` and effective AMP.
9. Each epoch appends a sample-weighted `EPOCH` record via
   `reduce_epoch`, including a `max_steps`-truncated epoch.
   `max_steps` caps successful global optimizer steps.
10. At `eval_interval` epochs, the final epoch, and any truncated
    terminal epoch, `evaluate_split` scores canonical TRAIN and VAL with
    the training objective (no capture, no bootstrap). Curves and strata
    are cleared for compact history; metrics, support, and warnings are
    retained. A null or absent selection metric fails the run.
11. `last.pt` is written at every evaluated epoch with kind, arch,
    state_dict, epoch, step, samples_seen, conditioning, config,
    provenance, dataset hash, and the selection metric, direction,
    value, and unbound validation evidence. The file checksum binds the
    recorded `CheckpointIdentity` and the train/val `checkpoint_hash`.
    A strict improvement per `is_improvement` byte-copies last to
    `best.pt` and keeps the earliest best on ties; `patience` counts
    non-improving evaluations and stops early. An `EVALUATION` record
    carrying the actual last identity and the improved flag is appended.
12. `execution.json` ends COMPLETED with stop_reason `epochs`,
    `max_steps`, or `early_stopping`; `best`, `last`, and `selected`
    follow `cfg.selected_checkpoint`. Failures write FAILED and
    `KeyboardInterrupt` writes INTERRUPTED; earlier history and
    checkpoints are preserved and no artifacts are erased.
13. `resources.json` persists `measure_resources` records for each
    unique train spatial shape at batches 1 and `cfg.batch_size`, with
    an explicit `error` entry when the partial counter does not support
    the model. The PARTIAL coverage marker is kept. Measurement runs on
    the final in-memory last-evaluated model state, so the payload also
    carries `checkpoint` (the `CheckpointIdentity` of `last.pt`, or null)
    and `model_state` set to `last_evaluated_checkpoint`; this may differ
    from the selected best checkpoint.
14. `cfg.checkpoint_path` receives a byte copy of the selected checkpoint:
    the copy is staged in a unique private temp inside the destination
    directory and published with an exclusive `os.link`, so an existing
    destination cannot be overwritten and later run changes never alter
    the published file. The run completes only after the copy succeeds.

## Errors and faults

`Err` on invalid configuration paths, dataset or availability failures,
refused destinations, non-finite loss or gradients without AMP,
evaluation failures, and persistence errors. `KeyboardInterrupt`
returns `Err` after persisting INTERRUPTED. If persisting the failed
status itself fails, both messages are returned.

## Messages

None.

## Configuration

`TrainConfig`; see [`tools.ml_models.train.config`](config.md).
`val_metric` resolves through `validation_metric` aliases before
selection.

## Constraints

- `torch`, the architecture registry, the loader, losses, and the
  evaluator are imported lazily inside the run; importing this module
  never requires torch.
- Checkpoint `validation` stored inside the payload carries no
  `checkpoint_hash`; the durable record binds it after checksum.
- There is no resume support; an interrupted run is terminal evidence.

## Related documents

- [`tools.ml_models.train`](../train.md)
- [`tools.ml_models.train.config`](config.md)
- [`tools.ml_models.analysis.training`](../analysis/training.md)
