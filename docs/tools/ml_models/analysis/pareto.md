# tools.ml_models.analysis.pareto

**Source:** `packages/tools/src/tools/ml_models/analysis/pareto.py`
**Kind:** module

## Purpose

This module holds the pure cost-versus-quality Pareto helpers: frontier
selection, the quality knee, seed collapsing, and table rendering. The
run-reader boundary `frontier_points` is unavailable until the catalog
reader lands.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `FrontierPoint` | dataclass | Run row projected to cost and score fields |
| `frontier_points` | function | Run-directory reader boundary; returns `Err` while unimplemented |
| `mean_by_arch` | function | Per-architecture mean across seeds |
| `pareto_front` | function | Non-dominated points ordered by cost |
| `knee` | function | Cheapest point within `spread` of the baseline holder |
| `knee_neighbors` | function | Contiguous front slice around the knee |
| `score_spread` | function | Seed-to-seed score range for one arch |
| `substitute_arch_placeholder` | function | Architecture placeholder substitution in space TOML |
| `orient_score` | function | Negate minimized metrics for comparisons |
| `format_pareto` | function | Front rendered as a table |

## Inputs and outputs

`frontier_points(runs, metric, cost_key=.., kind=.., split=..,
run_ids=..) -> Result[tuple[FrontierPoint, ..], str]`. The pure
helpers consume `FrontierPoint` tuples and return points, floats, or
text.

## Behavior

`frontier_points` returns `Err` for an unknown cost key or split and an
explicit unavailable `Err` on every call. `pareto_front` keeps
non-dominated points, ties broken by the first-seen cheaper arch. `knee`
rejects an empty front, a negative spread, or a missing baseline holder.

## Errors and faults

`frontier_points` returns `Err` on every call. `knee` and
`knee_neighbors` raise `ValueError` on invalid inputs.
`substitute_arch_placeholder` raises `ValueError` on empty, missing, or
ambiguous placeholders.

## Messages

None.

## Configuration

None.

## Constraints

- Frontier comparisons use one explicit split only.
- The reader boundary fails closed; no point is fabricated.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.runs`](runs.md)
