# tools.ml_models.export

**Source:** `packages/tools/src/tools/ml_models/export/`
**Kind:** package

## Purpose

The export package turns a trained conditioned checkpoint into a two-input
`(image, gsd)` ONNX artifact with a schema-2 JSON sidecar, validates artifact
bytes and declared graph shapes before any inference, runs finished-dataset
acceptance evaluation, and emits a combined classifier/segmentor pair
manifest only for accepted artifacts. Nothing here copies to active
deployment files.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`contract`](export/contract.md) | module | Conditioning markers and required GSD coverage |
| [`manifest`](export/manifest.md) | module | `ModelManifest` sidecar schema and JSON helpers |
| [`export`](export/export.md) | module | `ExportConfig` and two-input ONNX tracing |
| [`session`](export/session.md) | module | Hash-first session opening and I/O validation |
| [`accept`](export/accept.md) | module | Exhaustive test-split acceptance gate |
| [`pair`](export/pair.md) | module | Promotability and pair-manifest gates |
| [`calibration`](export/calibration.md) | module | Two-input INT8 calibration batches |
| [`precision`](export/precision.md) | module | FP16 and INT8 artifact conversions |

## Package interface

`tools.ml_models.export.__init__` carries a module docstring only. Callers
import the leaf modules. onnx, onnxruntime, and torch are lazy imports;
importing the package never requires them.

## Interactions

`contract` reads `flight.libs.config` and
`flight.payload.gimbal.footprint`/`intersect` geometry, and re-exports
`verify_conditioned_shapes` from `flight.payload.inference.contract`.
`export` builds models through `tools.ml_models.arch.registry` and reads
nested training provenance from checkpoints written by
`tools.ml_models.train.loop`. `accept` evaluates on finished datasets through
`tools.ml_models.train.evaluate`. `session` uses onnxruntime lazily.

## Constraints

- Only `pactnet`/`dilatenet` conditioned checkpoints export; every other
  architecture remains training-only.
- Existing artifacts and sidecars are never overwritten.
- Export and pairing do not copy to active or rollback deployment paths.
- Acceptance writes a sibling `.acceptance.json` report tied to the artifact
  SHA-256; `pair` refuses missing, failed, or mismatched evidence.

## Related documents

- [tools.ml_models](ml_models.md)
- [flight.payload.inference.contract](../flight/payload/inference/contract.md)
