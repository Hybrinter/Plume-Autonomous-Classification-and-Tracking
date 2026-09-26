# tools.ml_models.train.config

**Source:** `packages/tools/src/tools/ml_models/train/config.py`
**Kind:** module

## Purpose

This module holds frozen train hyperparameters, TOML overlay, the experiment
digest, and the pack channel-count rule.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TrainConfig` | class | Frozen train hyperparameters |
| `load_train_config` | function | Defaults plus an optional TOML overlay |
| `overlay_train_config` | function | Apply CLI field overlays |
| `apply_train_mapping` | function | Overlay from a string-key mapping |
| `config_digest` | function | 8-hex identity of experiment fields |
| `resolve_train_channels` | function | Channel count shared by the config and the pack |
| `write_train_config_toml` | function | Write a TrainConfig table |

## Inputs and outputs

`load_train_config(path=None) -> TrainConfig`.

`overlay_train_config(cfg, ...) -> TrainConfig`.

`apply_train_mapping(cfg, data) -> TrainConfig`.

`config_digest(cfg) -> str`.

`resolve_train_channels(cfg, pack) -> int`. The return value equals
`cfg.in_channels` and `pack.meta.in_channels`.

`write_train_config_toml(path, cfg) -> None`.

## Behavior

1. `TrainConfig` defaults to a segmentor, 3 bands, a 256 px crop, and one SGD
   epoch. `canvas` defaults to `None`. `max_steps` defaults to `None`.
2. `load_train_config` starts from those defaults. A TOML file overlays known
   keys. A `[canvas]` table maps onto `CanvasConfig`.
3. `overlay_train_config` replaces only the fields whose CLI value is not
   `None`.
4. `config_digest` hashes every field except `run_dir`, `run_id`,
   `checkpoint_path`, and `overwrite`.
5. `resolve_train_channels` reads `pack.meta.in_channels` and the image channel
   axis. Those two counts must match. `cfg.in_channels` must match them too.
   The function returns that count.
6. `write_train_config_toml` omits fields whose value is `None`. A set canvas
   is a `[canvas]` table. `frame_hw` is a two-integer array.

## Errors and faults

`ValidationError` on an unknown key or a field that fails the schema.
`ValueError` when pack metadata disagrees with the image tensor, or when
`in_channels` disagrees with the pack. `OSError` or `TOMLDecodeError` on a
missing or malformed TOML file.

## Messages

None.

## Configuration

`TrainConfig` defaults: `kind=segmentor`, `arch=""`, `input_height_px=256`,
`input_width_px=256`, `in_channels=3`, `epochs=1`, `batch_size=2`,
`learning_rate=0.01`, `momentum=0.9`, `weight_decay=0.0`, `optimizer=sgd`,
`scheduler=none`, `shuffle=false`, `pos_weight=0.0`, `augment=false`,
`loss=bce`, `focal_gamma=2.0`, `focal_alpha=0.25`, `amp=false`, `patience=0`,
`eval_interval=1`, `max_steps` unset, `canvas` unset,
`run_dir=artifacts/runs`, `overwrite=false`.

## Constraints

`in_channels` is the band count passed to the model. A value of 3 matches a
3-band pack. A 4-band pack needs `in_channels = 4`.

## Related documents

- [`tools.ml_models.train`](../train.md)
- [`tools.ml_models.train.loop`](loop.md)
- [`tools.ml_models.data.canvas`](../data/canvas.md)
- [`tools.ml_models.data.pack`](../data/pack.md)
