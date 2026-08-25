# tools.analysis.stats

**Source:** `packages/tools/src/tools/analysis/stats.py`
**Kind:** pure module

## Purpose

Stats reduces a `CaptureResult` to one summary row per emitted column. Numeric and categorical
signals share one tidy DataFrame for report tables.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `STATS_COLUMNS` | constant | Column order for the stats frame |
| `summarize` | function | Reduce capture to per-signal stats DataFrame |

## Inputs and outputs

**`summarize(capture) -> pd.DataFrame`**

- Input: `CaptureResult` with per-group wide frames.
- Output: DataFrame with columns in `STATS_COLUMNS`, one row per signal (registry order
  grouped by app).

## Behavior

1. `_column_meta` maps each emitted column (including `.cumulative` derivations) to group,
   unit, and kind.
2. Numeric columns get count, NaN count, mean, std, min, max, first, last, sum, unique
   count, and transition count.
3. Categorical columns get count, mode, first, last, unique count, and transition count.
4. Stats that do not apply to a kind are left as NaN or empty string sentinels.

## Errors and faults

None.

## Messages

None.

## Configuration

None.

## Constraints

- Computed entirely from recorder wide frames. Inherits run determinism.
- First and last numeric samples format as compact strings for Parquet uniformity.

## Related documents

- [`tools.analysis`](analysis.md)
- [`tools.analysis.recorder`](recorder.md)
- [`tools.analysis.report`](report.md)
