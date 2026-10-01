# tools.ml_models.analysis.runs

**Source:** `packages/tools/src/tools/ml_models/analysis/runs.py`
**Kind:** module

## Purpose

This module discovers training run directories, loads their
`summary.json` plus optional `eval.json` overlays, and renders list,
compare, and rank tables.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `discover_runs` | function | Run directories holding `summary.json`, sorted |
| `load_summary` | function | One run row: summary fields plus `test_*` overlay |
| `format_list` | function | One line per run directory |
| `format_compare` | function | Side-by-side table of run rows |
| `rank_runs` | function | Rows ordered by validation metric then FLOPs |
| `format_rank` | function | Ranked table of run rows |

## Inputs and outputs

Functions take run `Path` values and return plain dicts or formatted
strings.

## Behavior

`discover_runs` lists immediate children with a `summary.json`.
`load_summary` merges `summary.json` with `eval.json` keys prefixed
`test_`. `rank_runs` orders by the named metric (higher first) and breaks
ties on FLOPs.

## Errors and faults

`load_summary` raises `FileNotFoundError` when `summary.json` is absent.
Malformed JSON surfaces as `json.JSONDecodeError`.

## Messages

None.

## Configuration

None.

## Constraints

- Run discovery is one directory level deep.
- Ranking ignores runs missing the metric or a numeric FLOPs field.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.pareto`](pareto.md)
- [`tools.ml_models.train.loop`](../train/loop.md)
