# tools.ml_models.analysis.pareto

**Source:** `packages/tools/src/tools/ml_models/analysis/pareto.py`
**Kind:** module

## Purpose

This module builds a size-versus-quality frontier over the local run catalog.
Flight runs use parameter count against full-frame hit rate. Study and chip
runs use parameter count against validation Dice.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `COST_KEYS` | constant | `n_params` and `flops` |
| `SPLITS` | constant | `val` and `test` |
| `FULL_FRAME_HIT_RATE` | constant | Summary field `full_frame_hit_rate` |
| `STUDY_DICE_METRIC` | constant | Metric name `mean_dice` |
| `FrontierPoint` | class | One run reduced to score and cost |
| `frontier_points` | function | Points for an explicit metric and split |
| `has_full_frame_hit_rate` | function | Whether a summary carries the flight metric |
| `partition_catalog` | function | Flight runs, then study and chip runs |
| `flight_pareto` | function | Frontier on hit rate and `n_params` |
| `study_pareto` | function | Frontier on val Dice and `n_params` |
| `mean_by_arch` | function | Mean score across seeds of one architecture |
| `pareto_front` | function | Non-dominated points, cheapest first |
| `knee` | function | Cheapest point that holds a baseline |
| `knee_neighbors` | function | Knee plus neighbours on the frontier |
| `score_spread` | function | Max minus min score for one architecture |
| `substitute_arch_placeholder` | function | Fill an architecture placeholder in text |
| `orient_score` | function | Higher-is-better orientation |
| `format_pareto` | function | Text table of a frontier |

## Inputs and outputs

`frontier_points(runs, metric, cost_key="n_params", kind="", split="val", run_ids=None) -> tuple[FrontierPoint, ...]`.

`flight_pareto(runs) -> tuple[FrontierPoint, ...]`.

`study_pareto(runs) -> tuple[FrontierPoint, ...]`.

`format_pareto(front, metric, cost_key) -> str`.

## Behavior

1. `frontier_points` reads one split for every point. A run missing that score
   is dropped.
2. On `val`, `best_val_metric` applies only when `val_metric` names the same
   metric.
3. `full_frame_hit_rate` is also read from the unprefixed summary field.
4. `flight_pareto` keeps runs that carry `full_frame_hit_rate`. Cost is
   `n_params`. Score is that hit rate.
5. `study_pareto` keeps the other runs. Cost is `n_params`. Score is
   `val_mean_dice`.
6. `pareto_front` drops a point when another point is at least as good and no
   larger.
7. `substitute_arch_placeholder` edits text. It does not write a file.

## Errors and faults

`ValueError` on an unknown cost key or split, an empty frontier, a negative
spread, or a placeholder that cannot be replaced.

## Messages

None.

## Configuration

Default cost is `n_params`. Default split for `frontier_points` is `val`.

## Constraints

Torch-free. A frontier does not mix validation and test scores. The ml_models
CLI has no pareto command.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.runs`](runs.md)
- [`tools.ml_models.analysis.full_frame`](full_frame.md)
