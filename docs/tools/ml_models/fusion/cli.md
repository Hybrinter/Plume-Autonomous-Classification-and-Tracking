# tools.ml_models.fusion.cli

**Source:** `packages/tools/src/tools/ml_models/fusion/cli.py`
**Kind:** module

## Purpose

This module is the command line of the VIS/SWIR fusion trade.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `main` | function | Run the `test`, `late`, `thresholds`, or `seed-plot` command |

## Inputs and outputs

`main(argv=None) -> int`.

## Behavior

1. `test` scores the selected checkpoint of `--run` on the test split of
   `--dataset` (default: the training dataset). It writes `run`, `dataset`,
   and a `test` object of metric values.
2. `late` loads `--vis-run`, `--swir-run`, and `--reference-run` on
   `--dataset`, searches `--weights` on validation for `--metric`, and writes
   the chosen weight, the validation score per weight, and the `test` metrics.
3. `thresholds` reads `capture/val/rows.jsonl` and `capture/test/rows.jsonl`
   under `--analysis`. It writes the thresholds and the F1 rows, and a plot
   when `--plot` is given.
4. `seed-plot` reads `--metric` from the `test` object of each JSON file of
   each `--entry LABEL=a.json,b.json` and writes the seed plot.

## Errors and faults

Load errors, evaluation errors, and a bad `--entry` value exit with
`SystemExit` and a message.

## Messages

None.

## Configuration

`test` and `late` read a model analysis TOML through `--config`.

## Constraints

Do not run `test` or `late` while a training job runs on the same machine.

## Related documents

- [`tools.ml_models.fusion`](../fusion.md)
- [`tools.ml_models.fusion.study`](study.md)
- [`tools.ml_models.fusion.thresholds`](thresholds.md)
- [`tools.ml_models.fusion.plots`](plots.md)
- [`tools.ml_models.fusion.__main__`](__main__.md)
