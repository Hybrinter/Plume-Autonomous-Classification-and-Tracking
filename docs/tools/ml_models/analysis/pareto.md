# tools.ml_models.analysis.pareto

**Source:** `packages/tools/src/tools/ml_models/analysis/pareto.py`
**Kind:** module

## Purpose

This module builds the cost-versus-quality Pareto frontier over run
directories and selects the quality knee: the cheapest point that holds
the baseline score within a small spread.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `FrontierPoint` | dataclass | Run row projected to cost and score fields |
| `frontier_points` | function | Rows filtered by kind, split, and metric |
| `mean_by_arch` | function | Per-architecture mean across seeds |
| `pareto_front` | function | Non-dominated points ordered by cost |
| `knee` | function | Cheapest point within `spread` of the baseline holder |
| `knee_neighbors` | function | Contiguous front slice around the knee |
| `score_spread` | function | Seed-to-seed score range for one arch |
| `substitute_arch_placeholder` | function | `{arch}` substitution in table text |
| `orient_score` | function | Negate minimized metrics for comparisons |
| `format_pareto` | function | Front rendered as a table |

## Inputs and outputs

`frontier_points(rows, metric, split=...)` consumes `load_summary` rows;
`pareto_front`, `knee`, and `format_pareto` work on `FrontierPoint`
tuples.

## Behavior

`frontier_points` drops rows missing the metric, split, or cost fields and
negates minimized metrics. `pareto_front` keeps non-dominated points,
ties broken by the first-seen cheaper arch. `knee` rejects an empty front,
a negative spread, or a missing baseline holder.

## Errors and faults

`frontier_points` raises `ValueError` on an unknown split or cost key.
`knee` and `knee_neighbors` raise `ValueError` on invalid inputs.
`substitute_arch_placeholder` raises `ValueError` on empty, missing, or
ambiguous placeholders.

## Messages

None.

## Configuration

None.

## Constraints

- Frontier comparisons use the requested split only.
- Ranking uses the best validation metric for the matching metric name.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.runs`](runs.md)
