# tools.ml_models.cli

**Source:** `packages/tools/src/tools/ml_models/cli.py`
**Kind:** module

## Purpose

This module is the `python -m tools.ml_models` command line. It builds a
finished dataset from a raw tile source.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SourceName` | StrEnum | `flight`, `synthetic`, or `zenodo` raw source kinds |
| `app` | Typer application | `tools.ml_models` command group |
| `dataset_app` | Typer application | `dataset` subgroup under `app` |
| `build_command` | function | `dataset build` command |
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

`main(argv=None) -> int` returns a process exit code.

## Behavior

1. Load `BuildSpec` from `--spec`, or use the defaults.
2. `flight` requires `--source-dir` and calls `build_flight`.
3. `zenodo` requires `--images-tar`, `--labels-tar`, and `--weights-path`,
   then calls `build_zenodo`. `--bin-id` selects named bins from
   `DEFAULT_BINS`; an unknown or repeated name is rejected.
4. `synthetic` calls `build_synthetic` with `n` and `seed`.
5. `main` runs `app` under the program name `tools.ml_models` and converts
   `SystemExit` to an integer code.

## Errors and faults

`typer.BadParameter` when `--source flight` lacks `--source-dir`, when
`--source zenodo` lacks any required archive or weight option, or when
`--bin-id` is unknown or repeated. Filesystem and contract failures
(`OSError`, `ValueError`) from the spec load or the build surface as
`typer.BadParameter`, so a bad spec, source directory, archive, or
destination prints a concise parameter error message. Build error cases
are listed under
[`tools.ml_models.dataset.build`](dataset/build.md).

## Messages

None.

## Configuration

An optional `BuildSpec` TOML file; see
[`tools.ml_models.dataset.spec`](dataset/spec.md).

## Constraints

The command writes a new directory and refuses an existing one. The root
tools CLI mounts this application as `ml-models`.

## Related documents

- [`tools.ml_models`](../ml_models.md)
- [`tools.ml_models.dataset`](dataset.md)
- [`tools.ml_models.dataset.build`](dataset/build.md)
- [`tools.ml_models.dataset.spec`](dataset/spec.md)
- [`tools.cli`](../cli.md)
