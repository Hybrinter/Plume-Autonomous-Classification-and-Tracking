# tools.ml_models.analysis.metrics.generalization

**Source:** `packages/tools/src/tools/ml_models/analysis/metrics/generalization.py`
**Kind:** module

## Purpose

Frozen captured-row strata, train-only baselines, and recorded-group
bootstrap. Whole groups, not pixels or rows, are resampled with related
variants intact. Group IDs do not establish statistical independence.
Unsupported original metadata and compact-capture detail remain
unavailable. Every derived interval and stratum is persisted by callers
once; renderers never resample it.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `IntervalAudit` | dataclass | Declared replicate budget, actual valid/invalid support, and explicit unavailability |
| `BootstrapEvidence` | dataclass | Point metrics and intervals plus the replicate audit; no replicate arrays retained |
| `BaselineSpec` | dataclass | Fixed train-only constant classifier probability or all-background segmentation policy |
| `StratumIntervals` | dataclass | Audit keyed to the same named nullable cohort as its stratum metrics |
| `GeneralizationEvidence` | dataclass | Frozen cohort/stratum intervals, fixed baseline comparisons, and explicit coverage caveats |
| `MetricReference` | dataclass | Known metric definition or explicit unknown-name state; no inferred formula/direction |
| `ComponentObservation` | dataclass | One truth component joined to its parent capture/group, never a new independent sample |
| `DevelopmentGap` | dataclass | Explicit train-minus-validation scalar difference with both populations retained |
| `DevelopmentEvidence` | dataclass | Comparable dataset/scoring populations bound by the caller to one selected checkpoint |
| `score_captured_rows` | function | Exact supported scalar metrics from one identity-consistent captured cohort |
| `group_bootstrap` | function | Seeded percentile intervals by resampling whole recorded groups |
| `fit_baseline` | function | Fit the fixed policy on the canonical train cohort only |
| `measure_generalization` | function | Freeze recorded/truth strata, grouped intervals, and baseline comparisons once |
| `metric_reference` | function | Sorted unique definition lookups; unknown names keep their error |
| `development_gaps` | function | Train-minus-validation scalar gaps bound to one checkpoint hash |

## Inputs and outputs

`score_captured_rows(rows, score) -> Result[tuple[MetricValue, ...], str]`
takes captured `CaptureRow` values sharing one task, split, and dataset
identity.

`group_bootstrap(rows, score, cfg) -> Result[BootstrapEvidence, str]`
returns point metrics with optional `ConfidenceInterval` values plus the
per-metric `IntervalAudit` records.

`fit_baseline(rows) -> Result[BaselineSpec, str]` accepts canonical train
rows only.

`measure_generalization(rows, score, cfg, *, baseline=None) ->
Result[GeneralizationEvidence, str]` returns the frozen evidence record,
including the captured dataset hash, optional manifest hash, task,
split, both supplied configs, `outputs` availability records, and fixed
coverage `warnings`.

`metric_reference(names) -> tuple[MetricReference, ...]` is infallible:
each entry carries either the `MetricDefinition` or the lookup error.

`development_gaps(train, validation, score, *, checkpoint_hash) ->
Result[DevelopmentEvidence, str]` returns per-metric gaps with both
population metrics retained.

## Behavior

`score_captured_rows` validates identity consistency, unique canonical
keys, nonblank groups, and nonconflicting recorded-observation identity,
then orders rows by source key before reduction. Classifier cohorts are
reduced by `score_classifier` at the configured operating threshold.
Segmentor cohorts are reduced from the captured explicit-mask scalars
and require complete frozen `SpatialRow` provenance whose recorded
mask/blob/tolerance settings equal the supplied `ScoreConfig`; a
mismatch or a missing spatial row returns an actionable `Err` telling
the caller to re-analyze the checkpoint. Captured pixel counts are
checked against target/prediction support and unfiltered component
geometry. Spatial metrics come from `aggregate_spatial` over the frozen
rows.

`group_bootstrap` draws `bootstrap_replicates` whole-group samples with
replacement from a seeded generator. Replicates whose reduction produces
an unavailable metric value are counted per metric, not imputed. Fewer
than two recorded groups or fewer than two valid replicates leaves the
interval unavailable with an audit reason.

`measure_generalization` partitions rows into named nullable strata from
recorded metadata (`condition:*`, acquisition month), GSD fields
(`gsd_bin`, lateral/along bins, exact pairs, anisotropy, provenance),
and truth-image area bins — predicted size is never used — plus
truth-component size strata when complete spatial rows exist. Bootstrap
results are memoized by the exact ordered cohort keys, so a differently
named stratum over the same rows is not re-derived. A supplied
`BaselineSpec` is applied to the cohort without refitting. Availability
records mark grouped intervals, truth-component sizes, stratum pixel
ranking, the training baseline, conditions, and timestamps as
`AVAILABLE` or `UNAVAILABLE` with reasons.

`fit_baseline` fits only the canonical train split: classifiers record
the frozen train prevalence; segmentors use the all-background policy.
`development_gaps` compares `score_captured_rows` reductions for one
caller-bound `checkpoint_hash` between train and validation cohorts;
both populations must share task, dataset hash, and manifest identity,
and the returned record retains the dataset and manifest hashes, the
task, and the supplied `ScoreConfig`. Per-name differences are explicit
only when
unit, aggregation, and threshold agree, and the original supports and
unavailable reasons are retained.

## Errors and faults

Empty or identity-mixed cohorts, duplicate canonical keys, blank groups,
conflicting recorded-observation metadata, missing classifier logits,
incomplete or inconsistent segmentor captures, missing or mismatched
frozen spatial settings, non-train baseline fitting, invalid or
task-mismatched baseline specs, test/cross-dataset/leaking development
populations, and non-finite gaps all return `Err`.

## Messages

None.

## Configuration

`GeneralizationConfig` supplies `size_edges_px`, `size_edges_m2`,
`gsd_edges_m`, `bootstrap_replicates`, `confidence`, and `seed`. The
`ScoreConfig` supplied at measurement time must match the mask, blob,
and tolerance settings frozen in captured spatial rows.

## Constraints

- Recorded group IDs do not establish statistically independent
  observations; whole groups are resampled so related source/GSD
  variants and components travel together.
- Grouped-random split coverage is not a temporal or unseen-GSD
  holdout; intervals conditional on invalid class/support replicates
  report the valid count.
- Metre sizes and errors use local-GSD approximations; non-nominal GSD
  is not proof of calibration.
- Per-stratum pixel ranking histograms are not reconstructed from
  compact scalar captures.
- Baselines are fitted on canonical train only and are never refitted
  on held-out rows; development gaps never touch test.

## Related documents

- [`tools.ml_models.analysis.metrics`](../metrics.md)
- [`tools.ml_models.analysis.metrics.definitions`](definitions.md)
- [`tools.ml_models.analysis.metrics.spatial`](spatial.md)
- [`tools.ml_models.analysis.capture`](../capture.md)
