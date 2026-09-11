# tools.analysis.cli

**Source:** `packages/tools/src/tools/analysis/cli.py`
**Kind:** module

## Purpose

The CLI captures a named suite or scenario and writes a static analysis bundle. It also lists
available suites and scenarios.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `app` | Typer application | Analysis command group |
| `main` | function | Invoke `run` or `list` and return an exit code |

## Inputs and outputs

**`main(argv=None) -> int`**

- Input: optional argument vector (defaults to `sys.argv[1:]`).
- Output: process exit code (0 on success).

Usage:

```text
python -m tools.analysis run <suite|scenario> [--out DIR]
python -m tools.analysis list
```

## Behavior

1. The Typer application registers `run` and `list` subcommands.
2. `run` resolves specs via `suite_specs`, captures via `run_suite`, and writes via
   `write_suite_report`.
3. Default output directory is `artifacts/analysis/<name>`.
4. `run` prints scenario figure counts and registry signal totals.
5. `list` prints suite names from `suite_names` and scenario names from `scenario_names`.

## Errors and faults

Unknown suite or scenario names raise `KeyError` from characterize (uncaught, non-zero exit).

## Messages

None.

## Configuration

Optional `--out` overrides the bundle directory.

## Constraints

- Deterministic capture. No timestamps in output.
- Accepts meta-suites (`full`, `builtin`, `files`) and named groupings (`smoke`, `faults`,
  `commands`, `resources`).

## Related documents

- [`tools.analysis`](analysis.md)
- [`tools.analysis.characterize`](characterize.md)
- [`tools.analysis.report`](report.md)
