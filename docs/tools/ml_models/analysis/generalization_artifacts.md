# tools.ml_models.analysis.generalization_artifacts

**Source:** `packages/tools/src/tools/ml_models/analysis/generalization_artifacts.py`
**Kind:** module

## Purpose

Frozen generalization evidence serialization into bundle artifacts. The
codec turns an already-measured `GeneralizationEvidence` (and an
optional caller-bound `DevelopmentEvidence`) into checksummed
`BundleFile` bytes: canonical JSON documents, a flat typed metric
table, an interval audit table, and the table-schema manifest. It never
measures, resamples, reloads sources, or reads models or config; every
value is copied from the frozen records. The captured dataset hash,
manifest hash, task, split, and both scoring and generalization configs
travel inside `generalization.json` even when every interval is
unavailable.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `GENERALIZATION_METRICS_SCHEMA` | constant | Column contract for `tables/generalization_metrics.parquet` |
| `INTERVAL_AUDIT_SCHEMA` | constant | Column contract for `tables/interval_audit.csv` |
| `TABLE_SCHEMAS` | constant | Bundle path to schema mapping |
| `GeneralizationArtifacts` | dataclass | Returned file bytes and references |
| `generalization_artifacts` | function | `Result` codec boundary |

## Inputs and outputs

`generalization_artifacts(evidence, *, development=None, prefix="") ->
Result[GeneralizationArtifacts, str]` takes a frozen
`GeneralizationEvidence` and an optional `DevelopmentEvidence`, and
returns `files` (bundle bytes) plus `refs` (checksummed references). A
nonempty `prefix` must be a safe relative POSIX path; every returned
file path and reference path is placed under `prefix/...` while bytes,
checksums, row counts, and the relative paths inside the documents stay
unchanged. This lets several split namespaces coexist in one published
bundle without new measurement.

## Behavior

`generalization.json` is the canonical serialization of the supplied
evidence, including the frozen baseline spec and baseline curves.
`development.json` is emitted only when a development record is
supplied, and only when it shares the evidence's dataset hash, manifest
hash, scoring config, and task; a mismatch fails before staging.
`metric-reference.json` lists `metric_reference` lookups for every
metric name in the evidence plus development gap names; unknown names
keep their explicit lookup error without failing the codec.

`tables/generalization_metrics.parquet` carries one flat row per cohort,
stratum, and baseline metric with population/stratum provenance,
status/reason, support, threshold, and every interval field; absent
intervals leave all interval columns null. `tables/interval_audit.csv`
carries the verbatim replicate audits for cohort and stratum
populations. `n_attempted` is zero only when bootstrap resampling never
ran for the metric, such as a cohort with fewer than two recorded
groups; that is not a completed-replicate claim. Metrics whose
replicates did run but produced no valid reduction keep their
attempted, valid, and invalid counts unchanged. `tables/schema.json`
declares both column contracts.

Table codecs stage in a private temporary directory; the returned
bytes are read back into `BundleFile` and bound to `ArtifactRef`
checksums. The schema map keys and internal document paths stay
relative to the namespace root, so a prefixed artifact set matches its
unprefixed bytes exactly. No summary or output reservation happens
here.

## Errors and faults

Unsafe prefixes, incompatible development companions, codec,
serialization, and schema failures return `Err`; nothing is written
outside the private staging directory.

## Messages

None.

## Configuration

None.

## Constraints

- The writer copies frozen values; it never re-derives metrics,
  intervals, strata, or baselines.
- Baseline policy and curves stay inside `generalization.json`; no
  separately recomputed table exists for them.
- Unknown metric-reference entries remain explicit unknowns.
- A supplied `DevelopmentEvidence` must share the evidence's dataset,
  manifest, scoring, and task identity; the writer never rebinds them.
- `prefix` namespaces only the returned file and reference paths; it
  reserves no directories and changes no stored bytes.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.artifacts`](artifacts.md)
- [`tools.ml_models.analysis.metrics.generalization`](metrics/generalization.md)
