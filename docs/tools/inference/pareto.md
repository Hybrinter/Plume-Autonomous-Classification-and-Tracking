# tools.inference.pareto

**Source:** `packages/tools/src/tools/inference/pareto.py`
**Kind:** module

## Purpose

This module builds a size-versus-quality frontier over the local run catalog.
A run is on the frontier when no other run is at least as good on the metric
and no larger on the cost axis. Parameter count is the default cost. FLOPs are
available for latency comparisons.

Every point on one frontier must come from the same split. A catalog mixes runs
that were scored on validation alone with finalists that also carry a test
score. Using whichever field is present ranks a model on its test number against
rivals on validation. Test scores are the pessimistic ones. That comparison
penalises the runs that were evaluated most thoroughly.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `COST_KEYS` | constant | Selectable cost axes (`n_params`, `flops`) |
| `SPLITS` | constant | Selectable score splits (`val`, `test`) |
| `ARCH_PLACEHOLDER` | constant | Sentinel written into unfilled stage 2.5 and later spaces |
| `FrontierPoint` | dataclass | One run reduced to metric score and cost |
| `frontier_points` | function | Reduce run directories to comparable points |
| `mean_by_arch` | function | Collapse extra seeds of one architecture to a mean score |
| `pareto_front` | function | Non-dominated subset, ordered by cost |
| `knee` | function | Cheapest frontier point that holds a baseline score |
| `knee_neighbors` | function | Knee plus cheaper and costlier frontier neighbours |
| `score_spread` | function | Max minus min score of one architecture's seeds |
| `substitute_arch_placeholder` | function | Fill a space TOML arch placeholder |
| `orient_score` | function | Published metric to higher-is-better |
| `format_pareto` | function | Tab-separated text table of a frontier |

## Inputs and outputs

`frontier_points(runs, metric, cost_key="n_params", kind="", split="val",
run_ids=None) -> tuple[FrontierPoint, ...]`. Raises `ValueError` on an unknown
`cost_key` or `split`.

`mean_by_arch(points) -> tuple[FrontierPoint, ...]`.

`pareto_front(points) -> tuple[FrontierPoint, ...]`.

`knee(front, baseline_score, spread=0.0) -> FrontierPoint`. Raises `ValueError`
on an empty frontier, a negative spread, or no holding point.

`knee_neighbors(front, selected, beside=1) -> tuple[FrontierPoint, ...]`. Raises
`ValueError` when `beside` is negative or `selected` is not in `front`.

`score_spread(points, arch) -> float`.

`orient_score(metric, value) -> float`.

`substitute_arch_placeholder(text, arches) -> str`. Raises `ValueError` on an
empty name list, a missing placeholder, a multi-name scalar or in-list
assignment, or a placeholder that is not in an `arch` assignment.

`format_pareto(front, metric, cost_key) -> str`.

## Behavior

1. `frontier_points` reads each run with `load_summary`. When `kind` is set,
   runs with a different `kind` field are skipped. When `run_ids` is set, runs
   whose identifier is not in that set are skipped.
2. The `split` parameter names which score column to read: `val_<metric>` or
   `test_<metric>`. Every point is read from that split only. A run without that
   score is dropped.
3. On `val`, when `val_<metric>` is missing and the summary's `val_metric` field
   equals the requested metric, `best_val_metric` is used. A run swept on F1
   also stores that field; reading it as mean IoU would invent a number, so the
   guard applies only when the names match.
4. Metrics in `bce` and `brier` are negated so higher score is always better.
5. `mean_by_arch` groups points by kind and architecture. The score is the mean
   of the seeds. The path is the member with the median score.
6. `pareto_front` keeps points for which no other point has both a score at
   least as high and a cost no greater. The result is ordered by increasing
   cost. Ties on both axes keep the cheaper point, then the first seen.
7. `format_pareto` prints a header row and one row per frontier point, cheapest
   first. Minimized metrics are printed in their original orientation.
8. `knee` returns the cheapest frontier point whose oriented score is at least
   `baseline_score - spread`.
9. `knee_neighbors` returns a contiguous slice of the frontier around the knee.
   It keeps `beside` cheaper points and `beside` costlier points, and it
   truncates at the ends of the frontier.
10. `score_spread` returns the range of per-seed scores for one architecture.
    An architecture with fewer than two seeds returns 0.
11. `orient_score` negates `bce` and `brier`. Other metrics pass through.
12. `substitute_arch_placeholder` replaces the first list or scalar `arch`
    assignment that holds `ARCH_PLACEHOLDER`. A quoted placeholder that sits
    beside other names in an `arch` list is replaced in place and takes one
    name.

## Errors and faults

`ValueError` when `cost_key` is not `n_params` or `flops`, or when `split` is
not `val` or `test`. `knee` and `knee_neighbors` raise `ValueError` on empty
input, a negative allowance, a front that does not hold the baseline, or a
selected point that is not on the frontier. `substitute_arch_placeholder`
raises `ValueError` when the placeholder cannot be replaced.

## Messages

None.

## Configuration

Default cost axis is `n_params`. Default `kind` filter is empty (all kinds).
Default `split` is `val`, which is the split the search reads. Default knee
`spread` is 0. Default `beside` for `knee_neighbors` is 1.

## Constraints

Torch-free. Depends on `tools.inference.runs.load_summary`. Classifier and
segmentor runs must be filtered by `kind` before comparison. A frontier must
not mix validation and test scores.

## Related documents

- [`tools.inference.runs`](runs.md)
- [`tools.inference.cost`](cost.md)
- [`tools.inference`](../inference.md)
