# tools.ml_models.cli

**Source:** `packages/tools/src/tools/ml_models/cli.py`
**Kind:** module

## Purpose

This module is the `python -m tools.ml_models` command line. It builds a
finished dataset from a raw tile source, exposes the unavailable dataset
analysis, model analysis, and render boundaries, runs the unavailable
training boundary, exports two-input ONNX artifacts, gates acceptance,
writes classifier/segmentor pair manifests, and converts artifact
precision.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SourceName` | StrEnum | `flight` or `zenodo` raw source kinds |
| `app` | Typer application | `tools.ml_models` command group |
| `dataset_app` | Typer application | `dataset` subgroup under `app` |
| `build_command` | function | `dataset build` command |
| `dataset_analyze_command` | function | `dataset analyze` command |
| `train_command` | function | `train` command |
| `export_command` | function | `export` command |
| `accept_command` | function | `accept` command |
| `pair_command` | function | `pair` command |
| `convert_command` | function | `convert` command |
| `analyze_command` | function | `analyze` command |
| `render_command` | function | `render` command |
| `main` | function | Module entry point returning an exit code |

## Inputs and outputs

`dataset build` options:

- `--source`: `flight` or `zenodo`. Required.
- `--out`: finished dataset directory. Required.
- `--source-dir`: flight tile directory. Required with `--source flight`.
- `--images-tar`: Zenodo image archive. Required with `--source zenodo`.
- `--labels-tar`: Zenodo annotation archive. Required with `--source
  zenodo`.
- `--weights-path`: prism weight table TOML. Required with `--source
  zenodo`.
- `--bin-id`: Zenodo GSD bin name, repeatable. Selects `DEFAULT_BINS` by
  name (`native10` or `gsd15` through `gsd35`); default is every bin.
- `--spec`: optional `BuildSpec` TOML file.

`dataset analyze` options:

- `--dataset`: finished dataset directory. Required.
- `--out`: analysis output directory. Required.

`train` options:

- `--config`: optional `TrainConfig` TOML file; defaults apply when omitted.
- `--kind`: `classifier` or `segmentor`.
- `--arch`: architecture grammar name; empty selects the kind default.
- `--dataset`: finished dataset directory. Exactly one; a second
  occurrence is rejected.
- `--run-dir`: run root directory.
- `--run-id`: run directory name under `--run-dir`.
- `--device`: torch device; default is CUDA when available, else CPU.
- `--epochs`, `--batch-size`, `--max-steps`: loop controls.

`export` options:

- `--checkpoint`: trained conditioned checkpoint. Required.
- `--out`: destination ONNX artifact. Required.
- `--allow-partial-gsd`: accept short coverage and record the gap in the sidecar.
- `--dynamic-spatial` / `--no-dynamic-spatial`: varying image sizes for research,
  default off. Flight exports use the configured tile size. Dynamic spatial
  exports cannot pass flight promotion or pairing.

`accept` options:

- `--artifact`: ONNX artifact. Required.
- `--manifest`: model sidecar JSON. Required.
- `--dataset`: finished dataset directory. Exactly one. Required.
- `--min-iou`: segmentor per-source mean-IoU threshold, default 0.5.
- `--min-accuracy`: classifier per-source accuracy threshold, default 0.9.
- `--max-latency-ms`: worst batch-one CPU latency in milliseconds, default
  20.0. These timings are not the on-board 64-tile budget.

`pair` options:

- `--classifier-sidecar`, `--segmentor-sidecar`: model sidecars. Required.
- `--out`: destination pair manifest JSON. Required.
- `--allow-partial-gsd`: emit the manifest and record the coverage gap.

`convert` options:

- `--precision`: `fp16` or `int8`. Required.
- `--source`: source ONNX artifact with its sibling sidecar. Required.
- `--out`: new destination ONNX artifact. Required.
- `--dataset`: finished dataset directory; required for `int8` and
  refused for `fp16` beyond one occurrence.
- `--calib-samples`: INT8 calibration sample count, default 32.

`analyze` options:

- `--run`: training run directory. Required.
- `--out`: analysis output directory. Required.
- `--checkpoint`: checkpoint selector, default `best`.
- `--final-test` / `--no-final-test`: include the final-test evaluation,
  default off.

`render` options:

- `--evidence`: frozen evidence directory. Required.
- `--out`: destination figure directory. Required.

`main(argv=None) -> int` returns a process exit code.

## Behavior

1. Load `BuildSpec` from `--spec`, or use the defaults.
2. `flight` requires `--source-dir` and calls `build_flight`.
3. `zenodo` requires `--images-tar`, `--labels-tar`, and `--weights-path`,
   then calls `build_zenodo`. `--bin-id` selects named bins from
   `DEFAULT_BINS`; an unknown or repeated name is rejected.
4. `train` loads the optional `--config` TOML, applies the option overlay
   through `apply_train_mapping`, and calls `loop.train`. An `Ok` result
   echoes the run directory path; an `Err` becomes `typer.BadParameter`.
   The train modules import lazily inside the command.
5. `dataset analyze`, `analyze`, and `render` map their `Result`
   boundaries to `typer.BadParameter` on `Err`; while the boundaries are
   unavailable every invocation exits nonzero and creates no output.
6. `main` runs `app` under the program name `tools.ml_models` and converts
   `SystemExit` to an integer code.

## Errors and faults

`typer.BadParameter` when `--source flight` lacks `--source-dir`, when
`--source zenodo` lacks any required archive or weight option, or when
`--bin-id` is unknown or repeated. Filesystem and contract failures
(`OSError`, `ValueError`) from the spec load or the build surface as
`typer.BadParameter`, so a bad spec, source directory, archive, or
destination prints a concise parameter error message. `train` maps
configuration and run failures to `typer.BadParameter` the same way.
`export` and `pair` map `Err` results to `typer.BadParameter`. `accept`
maps an `Err` report to `typer.BadParameter`; while scoring is
unavailable no acceptance report is written and every call fails.
Build error cases are listed under
[`tools.ml_models.dataset.build`](dataset/build.md); run errors under
[`tools.ml_models.train.loop`](train/loop.md); export errors under
[`tools.ml_models.export.export`](export/export.md).

## Messages

None.

## Configuration

An optional `BuildSpec` TOML file for `dataset build`; see
[`tools.ml_models.dataset.spec`](dataset/spec.md). An optional
`TrainConfig` TOML file for `train`; see
[`tools.ml_models.train.config`](train/config.md).

## Constraints

The command writes a new directory and refuses an existing one. The root
tools CLI mounts this application as `ml-models`.

## Related documents

- [`tools.ml_models`](../ml_models.md)
- [`tools.ml_models.dataset`](dataset.md)
- [`tools.ml_models.dataset.build`](dataset/build.md)
- [`tools.ml_models.dataset.spec`](dataset/spec.md)
- [`tools.ml_models.train`](train.md)
- [`tools.ml_models.train.loop`](train/loop.md)
- [`tools.cli`](../cli.md)
