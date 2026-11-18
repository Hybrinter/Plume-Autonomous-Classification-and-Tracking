# tools.ml_models.fusion.thresholds

**Source:** `packages/tools/src/tools/ml_models/fusion/thresholds.py`
**Kind:** module

## Purpose

This module chooses classifier decision thresholds on the validation split and
scores them on the test split.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ScoredRows` | dataclass | Logits, labels, and look-angle bins of one split |
| `ThresholdChoice` | dataclass | One global logit threshold and one per bin |
| `ThresholdRow` | dataclass | Test F1 of one bin at the three thresholds |
| `ALL_BINS` | constant | Bin name `all` for the row over every tile |
| `read_capture_rows` | function | Read a capture `rows.jsonl` |
| `f1_score` | function | F1 of boolean predictions |
| `best_threshold` | function | Logit threshold with the highest F1 |
| `choose_thresholds` | function | Global and per-bin thresholds from validation rows |
| `score_thresholds` | function | Test F1 per bin and for all rows |

## Inputs and outputs

`read_capture_rows(path) -> ScoredRows`.

`choose_thresholds(val) -> ThresholdChoice`.

`score_thresholds(test, choice) -> tuple[ThresholdRow, ...]`.

## Behavior

1. Thresholds are logits. The fixed threshold is logit 0, which is probability 0.5.
2. `best_threshold` tests 197 candidates from the 1st to the 99th percentile of
   the split's logits. A tie keeps the lowest candidate.
3. `choose_thresholds` selects one threshold for all rows and one for each bin.
4. `score_thresholds` returns one row per test bin in sorted order, then `all`.
   A test bin with no validation rows uses the global threshold.

## Errors and faults

`best_threshold` raises `ValueError` on an empty split. `read_capture_rows`
raises `ValueError` for a row without a `logit` metric and `OSError` when the
file cannot be read.

## Messages

None.

## Configuration

None.

## Constraints

The functions other than `read_capture_rows` do no I/O.

## Related documents

- [`tools.ml_models.fusion`](../fusion.md)
- [`tools.ml_models.fusion.cli`](cli.md)
- [`tools.ml_models.fusion.plots`](plots.md)
