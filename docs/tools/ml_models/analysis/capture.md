# tools.ml_models.analysis.capture

**Source:** `packages/tools/src/tools/ml_models/analysis/capture.py`
**Kind:** module
**Status:** implemented

## Purpose

This module persists evaluator output under fixed memory and byte budgets.
The evaluator supplies scalar records and dense arrays for every scored
sample; the sink streams the scalar rows, keeps a bounded preview
selection, and optionally retains every dense prediction. Capture never
scores, labels, groups, bins, normalizes, or transforms any input.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `CaptureRow` | dataclass | Evaluator-supplied scalar record with selection flags |
| `CaptureSink` | Protocol | Runtime-checkable sink: `add`, `abort`, `close`, `references` |
| `BoundedCaptureSink` | class | Disk sink with bounded candidate cache and a byte budget |

`CaptureRow` is a frozen slots dataclass with fields `key`, `group_id`,
`bin_id`, `label`, `gsd_m`, `metrics`, `failure_score`, `false_positive`,
`false_negative`, `array_view`, and `spatial`. `label` must be exactly
0 or 1, `gsd_m` components must be strictly positive, `failure_score`
must be finite, and supplied `metrics` names must be unique.
`array_view` is the literal `CANONICAL`: arrays arrive in canonical
orientation regardless of the transform recorded in `key.element`.
`spatial` is an optional frozen `SpatialRow` carrying compact component
geometry, matches, unmatched identifiers, and boundary statistics for
segmentor rows — `None` for classifier rows and for rows persisted by
older writers. It retains no dense arrays. `metadata` carries the
recorded `ObservationMetadata` provenance copied from the stored row and
defaults to all-unavailable; `gsd_nominal` records whether the stored row
declared nominal GSD and stays `None` when the source schema cannot say.
`dataset_manifest_hash` carries the manifest digest the evaluator
verified against the on-disk `dataset.json`; when recorded it must be a
lowercase SHA-256 and stays `None` for rows persisted by older writers.
Rows persisted by older writers load all of these defaults unchanged.

`BoundedCaptureSink.create(out, cfg, dataset=...) -> Result` validates and
reserves the output; the direct constructor raises on the same failures.

## Inputs and outputs

`add(row, *, image, target, logits) -> Result[None, str]` accepts finite
float32 arrays with positive dimensions only. `image` is `(C, H, W)` with
`(H, W)` equal to `key.spatial_shard`, values in the unit interval
`[0, 1]`; `target` is binary. For `classifier` keys `target` and `logits`
are `(1,)`; for `segmentor` keys they are `(1, H, W)` aligned with the
image. Duplicate `SampleKey` digests fail.

`close() -> Result[None, str]` finishes selection and writes metadata.
`abort(reason) -> Result[None, str]` records a sticky failure without
removing the marker. `references() -> tuple[ArtifactRef, ...]` is empty
until `close` succeeds.

## Behavior

The output directory is created exclusively (`mkdir` refusing existing
targets) and must not overlap the resolved source dataset in either
direction. A `.incomplete` marker is written first and removed only on
successful close.

- `rows.jsonl`: one canonical JSON object per accepted row, appended and
  flushed incrementally (sorted keys, compact separators, `allow_nan`
  false). The complete row population is mandatory; budget exhaustion on
  scalar rows fails the sink.
- `previews/<key>.npz`: `image`, `target`, `logits` arrays written with
  `np.savez_compressed`, one file per selected key, no pickles.
- `full/<key>.npz`: under `retention="FULL"` only, one file per row
  written at `add` time; budget exhaustion here fails the sink.
- `metadata.json`: capture settings, `array_view`, `seen_rows`, per-family
  top-candidate key lists, selected key/family memberships with kept or
  omitted status, and the omitted count.

Preview selection keeps `examples_per_family` candidates per family:
REPRESENTATIVE ranks by the smallest seeded key digest, WORST ranks by
descending `failure_score` then digest, and FALSE_POSITIVE/FALSE_NEGATIVE
apply the same order to rows flagged by the evaluator. The digest is
SHA-256 of canonical JSON `{"seed": cfg.seed, "key": asdict(row.key)}`.
At close, families are merged round-robin in the fixed order
REPRESENTATIVE, WORST, FALSE_POSITIVE, FALSE_NEGATIVE, skipping duplicate
keys, capped at `max_preview_images`. The candidate array cache holds the
union of all family tops, at most `4 * examples_per_family` owned copies,
and evicts entries that fall out of every family.

The byte budget `max_capture_bytes` covers persisted payload plus cached
array bytes; the marker is excluded. A bounded-size metadata encoding is
reserved before any array caching or writing. Mandatory writes (scalar
rows and `FULL` arrays) evict optional cached candidates first, lowest
selection priority first, and fail only when the freed budget still does
not fit; family tops and scalar ordering never change. Selected
candidates that cannot be retained or written within the remaining
budget are recorded as `OMITTED` with reason `byte_budget` in the
metadata; scalar capture still completes. If the mandatory metadata
itself does not fit, `close` fails and the marker remains. The marker is
removed only after every file is written and its reference checksums
succeed, and the array cache is released on success, abort, and failure.

`references()` yields artifact refs built from actual file bytes:
`rows.jsonl` (population `evaluated_rows`), `metadata.json`
(`capture_metadata`), previews (`selected_previews`), and full files
(`full_predictions`).

## Errors and faults

All failures return `Err` with a failure message; any `add`/`close` failure is
sticky and keeps the marker. `abort` before `close` marks the sink failed
so a partial capture can never verify as complete; `close` then returns
the recorded reason. `close` is idempotent on success; `add` after close
fails.

## Messages

None.

## Configuration

`CaptureConfig` (`retention`, `max_preview_images`, `max_capture_bytes`,
`examples_per_family`, `seed`) from `analysis.config`.

## Constraints

- The protocol is `runtime_checkable`; all four methods are required.
- Only float32 finite arrays are accepted; no pickle or object arrays are
  written.
- Candidate memory is bounded independent of cohort size.
- Preview omission never changes which keys scientifically qualify; the
  omitted set is explicit in `metadata.json`.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.evaluate`](evaluate.md)
- [`tools.ml_models.analysis.config`](config.md)
