# tools.model.__main__

**Source:** `packages/tools/src/tools/model/__main__.py`
**Kind:** module

## Purpose

This module is the `python -m tools.model` entry. It parses `train`, `export`,
and `accept` subcommands.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `main` | function | Parse argv and dispatch. Returns a process exit code. |

## Inputs and outputs

`main(argv=None) -> int`. `argv` is the argument list without the program name.

## Behavior

1. Parse a required subcommand: `train`, `export`, or `accept`.
2. `train` overlays CLI flags onto `TrainConfig` and writes a checkpoint.
3. `export` writes an ONNX graph and a JSON manifest.
4. `accept` loads a manifest, runs the matching gate, and may `--promote`.
5. Return 0 on success, 1 on gate failure or a missing torch extra.

## Errors and faults

Argparse raises `SystemExit` on missing or unknown subcommands. Missing torch
prints an ImportError and returns 1.

## Messages

None.

## Configuration

Train flags: `--kind`, `--config`, `--data-dir`, `--out`, `--epochs`,
`--batch-size`, `--height`, `--width`, `--seed`.

Export flags: `--kind`, `--checkpoint`, `--out`, `--height`, `--width`,
`--version`, `--dataset-hash`, `--repo-sha`.

Accept flags: `--kind`, `--artifact`, `--manifest`, `--promote`, `--min-iou`,
`--min-accuracy`, `--max-latency-ms`, `--height`, `--width`.

## Constraints

The accept CLI runs live onnxruntime inference. Quality scenes are empty on the
CLI path; use the Python API with golden tensors for IoU or accuracy.

## Related documents

- [`tools.model`](../model.md)
- [`tools.model.train`](train.md)
- [`tools.model.export`](export.md)
- [`tools.model.accept`](accept.md)
