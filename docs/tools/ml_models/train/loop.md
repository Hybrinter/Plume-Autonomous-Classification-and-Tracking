# tools.ml_models.train.loop

**Source:** `packages/tools/src/tools/ml_models/train/loop.py`
**Kind:** module
**Status:** stub

## Purpose

This module is the public training boundary. Standard training is
unavailable until the evidence training phase lands.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `train` | function | Public `Result[Path, str]` boundary |

## Inputs and outputs

`train(cfg=None) -> Result[Path, str]`. `cfg=None` would use
`TrainConfig` defaults.

## Behavior

`train` returns `Err` with an explicit unavailable message on every call.
No run directory, model, dataset load, checkpoint, or summary is
created. The `ml-models train` CLI command maps the `Err` to a nonzero
exit and prints no run directory.

## Errors and faults

Always `Err`.

## Messages

None.

## Configuration

`TrainConfig`; see [`tools.ml_models.train.config`](config.md).

## Constraints

- The signature is stable for callers; the body creates no outputs.

## Related documents

- [`tools.ml_models.train`](../train.md)
- [`tools.ml_models.train.config`](config.md)
