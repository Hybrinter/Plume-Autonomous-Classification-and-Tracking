# tools.ml_models.train.loop

**Source:** `packages/tools/src/tools/ml_models/train/loop.py`
**Kind:** module

## Purpose

This module runs the training loop: one run directory, epoch batches from
the finished-dataset loader, periodic validation, checkpoints, and run
artifacts.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `train` | function | Public `Result[Path, str]` boundary |

## Inputs and outputs

`train(cfg=None) -> Result[Path, str]`. `Ok` carries the run directory;
`Err` carries the failure message. `cfg=None` uses `TrainConfig`
defaults; `dataset` must name one finished dataset root.

The run directory holds `config.toml`, `checkpoints/` (`best.pt` and
`last.pt`), `summary.json`, and `history.jsonl`.

## Behavior

1. `dataset` must name one finished dataset; an empty path fails before
   the run directory is created.
2. The run directory is `run_dir / (run_id or kind-digest-nanoseconds)`;
   an existing directory is rejected.
3. The manifest loads once and `training_provenance` runs once. The
   dataset must carry train and validation rows for the task.
4. The model comes from `arch.registry.build`; the optimizer is SGD or
   AdamW with the configured rates; `cosine` schedules `T_max=epochs`.
5. Each epoch draws `ceil(train_samples / batch_size)` batches via
   `make_loader` at seed `cfg.seed + epoch`.
6. Every batch calls `model(image, gsd)`; output shape must equal the
   target shape and a non-finite loss aborts the run before backward or
   the optimizer step. `amp` applies a CUDA `GradScaler` only on a CUDA
   device.
7. `evaluate` runs at `eval_interval`, the final epoch, and a
   `max_steps` stop. `val_metric` defaults to `f1` (classifier) or
   `mean_iou` (segmentor); `bce` and `brier` minimize, the rest
   maximize. `checkpoints/best.pt` follows the validation
   metric only; `patience` counts evaluations without improvement.
8. Checkpoints store kind, arch, `state_dict`, epoch, conditioning
   (`film-log-gsd-v1` for the conditioned families, `ignored` for
   wrapped graphs), config, provenance, and `dataset_hash`.
9. `summary.json` joins the checkpoint metadata with the last
   evaluation report; `history.jsonl` holds one record per evaluation.
10. `checkpoint_path` copies `checkpoints/last.pt` to a new path and
    refuses an existing destination.

## Errors and faults

`OSError`, `ValueError`, and `RuntimeError` surface as `Err` at the
boundary: an existing run directory, incompatible or leaky datasets,
missing splits, a model/target shape mismatch, a non-finite loss, or an
existing checkpoint destination. `overwrite` is reserved and never makes
a run destructive.

## Messages

None.

## Configuration

All controls come from `TrainConfig`; see
[`tools.ml_models.train.config`](config.md).

## Constraints

Sources own pixel preparation, the dataset build owns augmentation, and
the loader owns sampling and GSD encoding. The training loop adds no
pixel conversion. `torch.manual_seed(cfg.seed)` fixes initialisation and
batch order.

## Related documents

- [`tools.ml_models.train`](../train.md)
- [`tools.ml_models.train.config`](config.md)
- [`tools.ml_models.train.evaluate`](evaluate.md)
- [`tools.ml_models.train.losses`](losses.md)
- [`tools.ml_models.train.provenance`](provenance.md)
- [`tools.ml_models.dataset.loader`](../dataset/loader.md)
- [`tools.ml_models.arch.film`](../arch/film.md)
