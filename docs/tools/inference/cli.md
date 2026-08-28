# tools.inference.cli

**Source:** `packages/tools/src/tools/inference/cli.py`
**Kind:** module

## Purpose

The inference CLI runs training, evaluation, reporting, export, acceptance, and
dataset fetch workflows.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `app` | Typer application | Inference command group |
| `main` | function | Invoke the command group and return an exit code |
| `InferenceKind` | enum | Classifier and segmentor command choices |

## Inputs and outputs

`main(argv=None) -> int` accepts an optional argument vector without the program
name. Commands print generated paths, tables, acceptance details, or dataset
status.

## Behavior

1. `train` overlays options on `TrainConfig` and writes a run directory.
2. `eval` scores a checkpoint on a named split. The default split is `val`.
3. `report` writes figures and `report.md` into a run directory.
4. `list` prints local run summaries. `compare` prints a side-by-side table.
   `rank` prints the same columns in val-metric order.
5. `sweep` expands a space TOML, trains each trial, scores val, and writes
   JSONL. `arches` prints registered kind/name pairs.
6. `export` writes an ONNX graph and JSON manifest. `--int8` also writes a
   sibling INT8 QDQ pair.
7. `accept` runs the classifier or segmentor intake gate and optionally promotes
   an accepted artifact.
8. `fetch` delegates dataset status, download, unpack, and labeled preprocess
   work to `tools.inference.fetch`.

## Errors and faults

Invalid command input returns the Click usage-error exit code. A failed train
(`ValueError` or `FileExistsError`) or a failed acceptance gate returns 1.
Missing optional export dependencies print the import error and return 1.

## Messages

None.

## Configuration

Commands expose the same train, export, accept, eval, and fetch options as their
library configuration objects and functions. `train` overlays every
`TrainConfig` field. `--overwrite` sets `overwrite` true. `--shuffle` and
`--augment` set those flags true.

## Constraints

- Command callbacks remain thin wrappers around importable library functions.
- The accept command uses empty quality-scene lists. Use the Python acceptance
  API with golden tensors for IoU or accuracy checks.
- Torch and ONNX imports stay behind their optional extras.

## Related documents

- [`tools.inference`](../inference.md)
- [`tools.inference.train`](train.md)
- [`tools.inference.eval`](eval.md)
- [`tools.inference.report`](report.md)
- [`tools.inference.runs`](runs.md)
- [`tools.inference.sweep`](sweep.md)
- [`tools.inference.export`](export.md)
- [`tools.inference.accept`](accept.md)
- [`tools.inference.fetch`](fetch.md)
