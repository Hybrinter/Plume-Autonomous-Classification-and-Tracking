# tools.ml_models.cli

**Source:** `packages/tools/src/tools/ml_models/cli.py`
**Kind:** module

## Purpose

The ml_models CLI trains a run, scores a split, exports ONNX, accepts an
artifact, and writes a flight pair blob.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `app` | Typer application | ml_models command group |
| `main` | function | Invoke the command group and return an exit code |
| `ModelKind` | enum | Classifier and segmentor command choices |

## Inputs and outputs

`main(argv=None) -> int` accepts an optional argument vector without the
program name. Commands print a run directory, an eval path, export paths, an
acceptance detail line, or the pair JSON path.

## Behavior

1. `train` overlays options on `TrainConfig` and prints the run directory.
   `--canvas` sets `CanvasConfig()` flight-frame defaults.
2. `eval` scores a checkpoint. The default split is `val`. `--split test`
   scores the test split.
3. `export` writes an ONNX graph and a JSON sidecar for a run. A
   flight-promotable run traces 1544 by 2064. `--override-spatial` uses
   `--height` and `--width` for a research checkpoint.
4. `accept` runs hash, I/O contract, and golden-scene checks. `--flight`
   selects `InferenceConfig` shapes. Without `--flight`, expected shapes come
   from the manifest.
5. `pair` reads two sidecars and writes the deploy blob. The command exits 1
   when a run is not flight-promotable or a sidecar shape does not match.

## Errors and faults

Invalid command input returns the Click usage-error exit code. A failed train,
export, acceptance gate, or pair write returns 1.

## Messages

None.

## Configuration

`train` overlays `TrainConfig` fields and optional `--canvas`. `accept` uses
`FaultConfig.inference_timeout_ms` as the default latency budget. `--flight`
selects `InferenceConfig` geometry.

## Constraints

- Command logic stays in library functions.
- `inference` remains a separate root command.
- `--override-spatial` is not required for a flight-promotable export.

## Related documents

- [`tools.ml_models`](../ml_models.md)
- [`tools.ml_models.export`](export.md)
- [`tools.ml_models.train`](train.md)
- [`tools.cli`](../cli.md)
- [`tools.ml_models.__main__`](__main__.md)
