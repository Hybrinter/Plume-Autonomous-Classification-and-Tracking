# tools.ml_models.analysis.metrics.inputs

**Source:** `packages/tools/src/tools/ml_models/analysis/metrics/inputs.py`
**Kind:** module

## Purpose

Shared input validation and stable numeric transforms for the pure binary
scoring cores. Inputs are aligned vectors or N-by-one columns; targets are
exactly binary and scores are finite numeric values, never strings or
booleans.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `FloatVector` | type alias | `numpy` float64 vector |
| `BoolVector` | type alias | `numpy` boolean vector |
| `binary_vectors` | function | Validates aligned scores and labels; `Result` boundary |
| `sigmoid` | function | Stable logistic transform of raw logits |
| `finite_mean` | function | Overflow-safe mean of a nonempty finite vector |

## Inputs and outputs

`binary_vectors` takes score and label array-likes plus a `probabilities`
flag and returns `Result[tuple[FloatVector, BoolVector], str]`. `sigmoid`
takes a `FloatVector` of logits and returns probabilities. `finite_mean`
takes a `FloatVector` and returns a float.

## Behavior

`binary_vectors` accepts one-dimensional arrays or N-by-one columns.
Scores must carry an integer, unsigned-integer, or float dtype; integer
scores that exceed exact float64 representation (above 2**53) are
refused. Labels must be boolean or numeric and exactly 0 or 1. Inputs
must be nonempty, aligned, and finite; `probabilities=True` additionally
requires scores in `[0, 1]`. No clipping is applied. `sigmoid` evaluates
the positive and negative branches separately for stability and leaves
the raw ranking scores unchanged. `finite_mean` accumulates with
`math.fsum` so a preliminary sum cannot overflow.

## Errors and faults

`binary_vectors` returns `Err` for wrong shapes, non-numeric scores,
non-binary targets, empty or misaligned inputs, non-finite values,
out-of-range probabilities, and over-precise integers.

## Messages

None.

## Configuration

None.

## Constraints

- Validation never clips, resamples, or reweights inputs.
- Raw ranking scores and displayed probabilities are kept separate.

## Related documents

- [`tools.ml_models.analysis.metrics`](../metrics.md)
