# tools.ml_models.analysis.metrics.calibration

**Source:** `packages/tools/src/tools/ml_models/analysis/metrics/calibration.py`
**Kind:** module

## Purpose

Pure probability calibration evidence with explicit empty-bin support.
Brier is an unweighted probability score. Equal-width reliability bins
are left-closed/right-open except the final closed bin, and ECE depends
on this binning.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ReliabilityBin` | dataclass | One bin with null measurements when its support is zero |
| `CalibrationEvidence` | dataclass | Metrics, exact bin aggregates, curves, and support |
| `score_calibration` | function | Brier, reliability, and ECE on aligned observations; `Result` boundary |

## Inputs and outputs

`score_calibration` takes probability and label array-likes plus a
keyword `n_bins` (integer, at least two) and returns
`Result[CalibrationEvidence, str]`.

## Behavior

Inputs are validated by `binary_vectors` with `probabilities=True`.
Brier is the unweighted mean of `(p - y) ** 2`. Equal-width bins cover
`[0, 1]`; each bin records its count, mean predicted probability, and
observed positive fraction, with null measurements when empty. ECE sums
`(n_bin / N) * abs(mean_probability - positive_fraction)` over occupied
bins only. The `calibration_reliability` curve plots observed fraction
against mean predicted probability, using the bin midpoint as the x
coordinate for empty bins; the `calibration_support` curve reports
per-bin counts at bin midpoints.

## Errors and faults

A non-integer or sub-two `n_bins` returns `Err`; invalid vectors
propagate `Err` from `binary_vectors`.

## Messages

None.

## Configuration

`n_bins` defaults to 10 when called directly; `score_classifier` passes
`ScoreConfig.n_calibration_bins`.

## Constraints

- ECE is bin-dependent; it is not a proper scoring rule or a substitute
  for the reliability curve.
- Empty bins contribute zero ECE weight and null, not zero, means.

## Related documents

- [`tools.ml_models.analysis.metrics`](../metrics.md)
- [`tools.ml_models.analysis.metrics.inputs`](inputs.md)
- [`tools.ml_models.analysis.metrics.classifier`](classifier.md)
