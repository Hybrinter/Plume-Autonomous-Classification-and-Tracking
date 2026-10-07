# tools.ml_models.analysis.contracts

**Source:** `packages/tools/src/tools/ml_models/analysis/contracts.py`
**Kind:** module

## Purpose

This module declares the strict frozen typed records that join evaluation,
capture, and rendering: the sample alignment key, support and interval
metadata, named measurements, curve evidence, artifact references, and the
identity records for datasets, code, and checkpoints.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `Task` | type alias | `classifier` or `segmentor` |
| `Split` | type alias | `train`, `val`, or `test` |
| `SupportUnit` | type alias | `IMAGE`, `PIXEL`, `COMPONENT`, or `GROUP` |
| `MetricStatus` | type alias | `AVAILABLE` or `UNAVAILABLE` |
| `CurveMethod` | type alias | `EXACT` or `HISTOGRAM` |
| `ArtifactKind` | type alias | `TABLE`, `CURVE`, `FIGURE`, `VISUAL`, `PREDICTIONS`, `CONFIG`, or `REFERENCE` |
| `AvailabilityStatus` | type alias | `AVAILABLE`, `UNAVAILABLE`, or `SKIPPED` |
| `is_sha256` | function | Lowercase 64-hex SHA-256 check |
| `check_bundle_path` | function | Safe relative POSIX path validation |
| `SampleKey` | dataclass | Dataset, task, split, shard, row, tile, and element identity |
| `NamedCount` | dataclass | One named subpopulation count |
| `MetricSupport` | dataclass | Support unit, size, and named counts |
| `ConfidenceInterval` | dataclass | Ordered finite interval with replicate metadata |
| `MetricValue` | dataclass | One named value or an explicit `UNAVAILABLE` status |
| `CurveEvidence` | dataclass | Named aligned finite coordinates with method metadata |
| `ArtifactRef` | dataclass | Content-addressed reference to a bundle file |
| `DatasetIdentity` | dataclass | Content and manifest hashes plus dataset metadata |
| `CodeIdentity` | dataclass | Revision/dirty provenance with explicit unknown state |
| `CheckpointIdentity` | dataclass | Checkpoint hash, task, arch, and training dataset |
| `AvailabilityRecord` | dataclass | One output's availability with reason and requirement |
| `StratumEvidence` | dataclass | Named cohort value with metrics and support |
| `SplitEvidence` | dataclass | Task/split/dataset identity plus metrics, curves, artifacts, and strata |
| `SourceSnapshot` | dataclass | Typed frozen source-file reference (path, hash, size) |

## Inputs and outputs

Constructor arguments only. All records validate at construction and raise
`ValueError` on violation. `check_bundle_path` takes a path string and
returns `None` or raises; `is_sha256` returns a `bool`.

## Behavior

The records are frozen, slots pydantic dataclasses with `extra="forbid"`.
Integer fields reject booleans and non-integers; float fields reject NaN
and infinity. An `AVAILABLE` metric requires a finite value and no reason;
an `UNAVAILABLE` metric requires a `None` value and a nonempty reason and
cannot carry an interval.
Curve `x`/`y` lengths must match; thresholds are empty or aligned, and a
`None` threshold may mark the predict-none operating point. `HISTOGRAM`
curves require `n_bins >= 2` and a descriptive note; `EXACT` curves cannot
carry `n_bins`. Artifact paths are safe relative POSIX paths. `UNAVAILABLE`
and `SKIPPED` availability records require reasons. `StratumEvidence`
requires a nonblank name, a nonblank value when present, and unique
metric names. `SplitEvidence` requires unique metric names, curve names,
artifact paths, and stratum `(name, value)` identities.

## Errors and faults

All violations raise `ValueError` at construction; no partial or normalized
record is produced.

## Messages

None.

## Configuration

None.

## Constraints

- Hashes are strict lowercase 64-hex SHA-256 strings.
- Counts, indices, sizes, and seeds are exact nonnegative integers where
  required; booleans are rejected.
- `DatasetIdentity.manifest_hash` is the SHA-256 of the actual
  `dataset.json` bytes, kept separate from the shard content hash.
- `CodeIdentity` requires a reason when revision or dirty state is absent;
  a clean state is never invented.
- No measurement, interval, or curve algorithm executes here; these are
  data contracts only.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.summaries`](summaries.md)
- [`tools.ml_models.analysis.artifacts`](artifacts.md)
