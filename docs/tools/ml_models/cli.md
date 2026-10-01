# tools.ml_models.cli

**Source:** `packages/tools/src/tools/ml_models/cli.py`
**Kind:** module

## Purpose

This module is the `python -m tools.ml_models` command line. It builds a
finished dataset from a raw tile source, trains models on finished
datasets, exports two-input ONNX artifacts, gates acceptance, and writes
classifier/segmentor pair manifests.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SourceName` | StrEnum | `flight`, `synthetic`, or `zenodo` raw source kinds |
| `app` | Typer application | `tools.ml_models` command group |
| `dataset_app` | Typer application | `dataset` subgroup under `app` |
| `build_command` | function | `dataset build` command |
| `train_command` | function | `train` command |
| `export_command` | function | `export` command |
| `accept_command` | function | `accept` command |
| `pair_command` | function | `pair` command |
| `main` | function | Module entry point returning an exit code |

## Inputs and outputs

`dataset build` options:

- `--source`: `flight`, `synthetic`, or `zenodo`. Required.
- `--out`: finished dataset directory. Required.
- `--source-dir`: flight tile directory. Required with `--source flight`.
- `--images-tar`: Zenodo image archive. Required with `--source zenodo`.
- `--labels-tar`: Zenodo annotation archive. Required with `--source
  zenodo`.
- `--weights-path`: prism weight table TOML. Required with `--source
  zenodo`.
- `--bin-id`: Zenodo GSD bin name, repeatable. Selects `DEFAULT_BINS` by
  name; default is every bin.
- `--spec`: optional `BuildSpec` TOML file.
- `--n`: synthetic tile count, default 12.
- `--seed`: synthetic image seed, default 0.

`train` options:

- `--config`: optional `TrainConfig` TOML file; defaults apply when omitted.
- `--kind`: `classifier` or `segmentor`.
- `--arch`: architecture grammar name; empty selects the kind default.
- `--dataset`: finished dataset directory, repeatable.
- `--dataset-weight`: sampling weight per dataset, repeatable.
- `--run-dir`: run root directory.
- `--run-id`: run directory name under `--run-dir`.
- `--device`: torch device; default is CUDA when available, else CPU.
- `--epochs`, `--batch-size`, `--max-steps`: loop controls.

`export` options:

- `--checkpoint`: trained conditioned checkpoint. Required.
- `--out`: destination ONNX artifact. Required.
- `--allow-partial-gsd`: accept short coverage and record the gap in the sidecar.
- `--dynamic-spatial` / `--no-dynamic-spatial`: dynamic H/W axes, default on.

`accept` options:

- `--artifact`: ONNX artifact. Required.
- `--manifest`: model sidecar JSON. Required.
- `--dataset`: finished dataset directory, repeatable; at least one.
- `--min-iou`: segmentor per-source mean-IoU threshold, default 0.5.
- `--min-accuracy`: classifier per-source accuracy threshold, default 0.9.
- `--max-latency-ms`: worst batch-one CPU latency in milliseconds, default
  20.0. These timings are not the on-board 64-tile budget.

`pair` options:

- `--classifier-sidecar`, `--segmentor-sidecar`: model sidecars. Required.
- `--out`: destination pair manifest JSON. Required.
- `--allow-partial-gsd`: emit the manifest and record the coverage gap.

`main(argv=None) -> int` returns a process exit code.

## Behavior

1. Load `BuildSpec` from `--spec`, or use the defaults.
2. `flight` requires `--source-dir` and calls `build_flight`.
3. `zenodo` requires `--images-tar`, `--labels-tar`, and `--weights-path`,
   then calls `build_zenodo`. `--bin-id` selects named bins from
   `DEFAULT_BINS`; an unknown or repeated name is rejected.
4. `synthetic` calls `build_synthetic` with `n` and `seed`.
5. `train` loads the optional `--config` TOML, applies the option overlay
   through `apply_train_mapping`, and calls `loop.train`. An `Ok` result
   echoes the run directory path; an `Err` becomes `typer.BadParameter`.
   The train modules import lazily inside the command.
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
writes the acceptance report and exits nonzero when a gate fails.
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
