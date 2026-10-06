# tools.ml_models.analysis.prediction_artifacts

**Source:** `packages/tools/src/tools/ml_models/analysis/prediction_artifacts.py`
**Kind:** module

## Purpose

Prediction preview and gallery serialization into bundle artifacts.
The codec preserves supplied preview NPZ files byte-for-byte and
emits a canonical `prediction-manifest.json` recording every bound
preview and every frozen `PredictionGallery`. No row is picked, no
identity is re-hashed from sources, and no capture or model is read.
Rendered figure references are bound separately by the bundle writer.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `PredictionArtifacts` | dataclass | Returned file bytes and references |
| `prediction_artifacts` | function | `Result` codec boundary |

## Inputs and outputs

`prediction_artifacts(captured, *, prefix="") ->
Result[PredictionArtifacts, str]` takes a
`PredictionPreviewCapture` and returns `files` (preview NPZ bytes plus
the manifest) and `references`. A nonempty `prefix` namespaces every
returned path under `prefix/...` without changing bytes or checksums.

## Behavior

`prediction-manifest.json` is the canonical serialization under
`{"schema_version": 1, "previews": [...], "galleries": [...]}` — each
preview keeps its bound row, path, checksum, size, and display
mapping, and each gallery keeps its chosen rows, availability record,
and selection method. Preview paths must be unique and resolve to
exactly one supplied file whose size and checksum match the recorded
identity. Cache NPZ references keep kind `REFERENCE`/format `npz`;
the manifest keeps `REFERENCE`/`json`.

## Errors and faults

Unsafe prefixes, duplicate preview paths or row keys, manifest path
collisions, supplied files that do not exactly match the preview
descriptors, missing file bytes, and size or checksum mismatches
return `Err` before caller output is touched.

## Messages

None.

## Configuration

None.

## Constraints

- The codec preserves frozen selections; it performs no re-selection,
  thresholding, or identity rehashing.
- `prefix` namespaces only returned paths; it reserves no directories.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.prediction_selections`](prediction_selections.md)
- [`tools.ml_models.analysis.visuals.predictions`](visuals/predictions.md)
