# tools.ml_models.train.config

**Source:** `packages/tools/src/tools/ml_models/train/config.py`
**Kind:** module

## Purpose

This module defines `TrainConfig`, the frozen training configuration for
finished-dataset runs, plus the TOML load/overlay helpers and the run
digest.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TrainConfig` | dataclass | Training paths and optimizer controls |
| `load_train_config` | function | Strict flat TOML load, or defaults |
| `apply_train_mapping` | function | Validated field overlay |
| `config_digest` | function | SHA-256 prefix over experiment fields |
| `write_train_config_toml` | function | Flat TOML writer |

## Inputs and outputs

`load_train_config(path=None) -> TrainConfig`. A path reads strict flat
TOML; unknown keys fail.

`apply_train_mapping(cfg, data) -> TrainConfig` merges a mapping over the
current config and revalidates.

`config_digest(cfg) -> str` hashes every field except `run_dir`,
`run_id`, `checkpoint_path`, and `overwrite`.

`write_train_config_toml(path, cfg)` writes one `key = value` line per
field, omitting None values.

## Behavior

1. `kind` is `classifier` or `segmentor`; `arch` empty selects the kind
   default.
2. `dataset` names the one finished dataset root; an empty string fails
   at run start. The legacy `datasets` and `dataset_weights` keys are
   rejected with a single-dataset message.
3. `epochs`, `batch_size`, and `eval_interval` are positive;
   `patience` is nonnegative; `max_steps` is None or positive.
4. `learning_rate` is finite and positive; `weight_decay` is finite and
   nonnegative; `momentum` lies in `[0, 1)`.
5. `optimizer` is `sgd` or `adamw`; `scheduler` is `none` or `cosine`.
6. `loss` is `bce`, `dice`, `bce_dice`, `focal`, or `focal_dice`;
   `focal_gamma` is nonnegative, `focal_alpha` lies in `[0, 1]`, and
   `pos_weight` is nonnegative.
7. `val_metric` is empty or a metric valid for the kind: classifier
   metrics include `f1`, `brier`, and `bce`; segmentor metrics include
   `mean_iou`, `mean_dice`, `mean_iou_blob_gate`, and `bce`.

## Errors and faults

`ValueError` (pydantic `ValidationError`) on unknown keys, out-of-bounds
values, the legacy `datasets` or `dataset_weights` keys, or a
`val_metric` invalid for the kind.

## Messages

None.

## Configuration

The dataclass fields are the configuration; TOML files mirror field names
one to one.

## Constraints

The dataclass is frozen and forbids extra keys. The digest excludes only
output-path controls.

## Related documents

- [`tools.ml_models.train`](../train.md)
- [`tools.ml_models.train.loop`](loop.md)
- [`tools.ml_models.cli`](../cli.md)
