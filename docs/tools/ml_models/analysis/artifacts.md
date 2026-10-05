# tools.ml_models.analysis.artifacts

**Source:** `packages/tools/src/tools/ml_models/analysis/artifacts.py`
**Kind:** module

## Purpose

This module owns the versioned evidence codecs, content-addressed artifact
references, dataset identity, exclusive bundle publication, bundle
verification, and typed CSV/Parquet tables for the analysis stack.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SUMMARY_FILENAME` | constant | `summary.json` |
| `INCOMPLETE_FILENAME` | constant | `.incomplete` publication marker |
| `Scalar` | type alias | `str`, `int`, `float`, `bool`, or `None` |
| `ColumnDtype` | type alias | `STRING`, `INT64`, `FLOAT64`, or `BOOLEAN` |
| `encode_summary` | function | Canonical sorted-key UTF-8 JSON bytes; `Result` boundary |
| `decode_summary` | function | Tagged summary parse; `Result` boundary |
| `checksum_file` | function | Streaming SHA-256 of a file; `Result` boundary |
| `dataset_identity` | function | Verified dataset identity from a root; `Result` boundary |
| `artifact_ref` | function | Reference factory deriving hash/size from bytes |
| `artifact_file_ref` | function | Reference factory deriving hash/size from a file |
| `BundleFile` | dataclass | One safe-path file slated for publication |
| `publish_bundle` | function | Exclusive bundle write; `Result` boundary |
| `verify_bundle` | function | Bundle verification returning the summary; `Result` boundary |
| `ColumnSpec` | dataclass | One named typed column with a nullability flag |
| `TableSchema` | dataclass | Ordered unique `ColumnSpec` tuple |
| `write_table` | function | Exclusive typed `.csv`/`.parquet` write; `Result` boundary |
| `read_table` | function | Schema-checked table read; `Result` boundary |

## Inputs and outputs

`encode_summary` takes a `Summary` and returns `Result[bytes, str]`.
`decode_summary` takes bytes and returns `Result[Summary, str]`.
`checksum_file` and `dataset_identity` take paths and return `Result`
values. `publish_bundle` takes an output `Path`, a `Summary`, a tuple of
`BundleFile`, and an optional `dataset_root`, and returns
`Result[Path, str]`. `verify_bundle` takes a bundle root and returns
`Result[Summary, str]`. `write_table` takes a destination `Path`, a tuple
of row dicts, a `TableSchema`, and an optional `bundle_path`, and returns
`Result[ArtifactRef, str]`. `read_table` returns
`Result[tuple[dict[str, Scalar], ...], str]`.

## Behavior

Summaries encode as sorted-key JSON with NaN/Infinity refused; decode
dispatches on `summary_kind` and rejects unknown kinds, versions, extra
fields, and non-exact counts. `dataset_identity` reads the actual
`dataset.json` bytes for `manifest_hash`, verifies shard content through
the manifest loader, and never writes the dataset. Only `COMPLETE`
summaries publish; `PARTIAL` and `FAILED` summaries are refused before
the output is reserved. Publication reserves the output directory with
`mkdir(exist_ok=False)`, writes the `.incomplete` marker first, writes
each file exclusively, writes the summary last, then removes the marker;
a failed write leaves the marker in place and fails verification. The
same `ArtifactRef` may appear at the root and in split or history
references; identical references deduplicate by path, while conflicting
references that share a path are refused before any output is made.
References must match supplied bytes exactly; extra unreferenced files
are refused. Verification rejects missing or corrupt references, unsafe
paths, and targets that escape the resolved bundle root through links or
junctions. CSV uses a canonical scalar encoding: `\N` is the only null
marker, a leading backslash escapes strings that start with one, empty
strings are never null, integers are decimal, floats use `repr`, and
booleans are `true`/`false`. Decoding removes only a doubled leading
backslash; a single leading backslash outside the null marker is
rejected. Parquet uses explicit Arrow schemas, and reads re-check names,
dtypes, and nullability.

## Errors and faults

Every public boundary returns `Err` for missing/corrupt input, existing
outputs, reference mismatches, escapes, and unsupported suffixes.
Constructor violations raise `ValueError`. A failed publication leaves
the reserved directory and marker in place; preexisting directories are
never deleted.

## Messages

None.

## Configuration

None.

## Constraints

- Only `COMPLETE` summaries publish or verify; `PARTIAL`/`FAILED`
  summaries are refused before reservation, and a bundle that still
  carries the marker fails verification.
- The output must not resolve inside a supplied source dataset root.
- The summary file is never one of its own referenced files; reserved
  bundle paths are refused.
- Reference factories derive checksums and sizes from actual bytes or
  files; no guessed values.
- No schema is inferred from a row, and no duplicate key or row is
  dropped.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.contracts`](contracts.md)
- [`tools.ml_models.analysis.summaries`](summaries.md)
