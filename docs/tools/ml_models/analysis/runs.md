# tools.ml_models.analysis.runs

**Source:** `packages/tools/src/tools/ml_models/analysis/runs.py`
**Kind:** module

## Purpose

This module holds the run-catalog boundary and text table formatters.
The catalog readers are unavailable until the evidence analysis phase
lands; the formatters remain pure over caller-supplied rows.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `discover_runs` | function | Run-directory catalog reader; returns `Err` while unimplemented |
| `load_summary` | function | Run-summary reader; returns `Err` while unimplemented |
| `rank_runs` | function | Stored-summary ranking; returns `Err` while unimplemented |
| `format_list` | function | One line per summary row |
| `format_compare` | function | Side-by-side table of summary rows |
| `format_rank` | function | Compare table in the caller's order |

## Inputs and outputs

`discover_runs(root) -> Result[tuple[Path, ..], str]`,
`load_summary(run_dir) -> Result[dict[str, object], str]`, and
`rank_runs(runs, metric) -> Result[tuple[dict[str, object], ..], str]`.
The formatters take `tuple[dict[str, object], ..]` rows and return text.

## Behavior

The three readers return `Err` with an explicit unavailable message on
every call. The formatters render the given rows and never read files.
No ranking fallback over supplied rows remains.

## Errors and faults

`discover_runs`, `load_summary`, and `rank_runs` always return `Err`.
The formatters do not raise on missing fields; absent cells render
empty.

## Messages

None.

## Configuration

None.

## Constraints

- Readers fail closed: no empty catalogs or fabricated rows.
- Formatters consume supplied rows only and perform no I/O.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.pareto`](pareto.md)
