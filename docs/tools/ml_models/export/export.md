# tools.ml_models.export.export

**Source:** `packages/tools/src/tools/ml_models/export/export.py`
**Kind:** module

## Purpose

This module exports a trained conditioned checkpoint as a dynamic-batch
two-input ONNX artifact plus a schema-2 sidecar, publishing
through private sibling temporaries and exclusive links so existing files —
including ones created concurrently after the precheck — are never
overwritten.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ExportConfig` | dataclass | `checkpoint_path`, `output_path`, `opset=17`, `dynamic_spatial=True`, `allow_partial_gsd=False` |
| `export` | function | `Result[Path, str]` export entry |

## Behavior

1. Rejects an existing artifact or sidecar before loading anything.
2. Loads the checkpoint with `torch.load(weights_only=True)`, requires the
   `film-log-gsd-v1` conditioning marker, and — before any tracing —
   requires provenance `norm` `unit`, the `BLUE`/`GREEN`/`RED` band list,
   and the default flight `gsd_reference_m`. It then builds the model from
   the nested provenance (`kind`, `arch`, `in_channels`) with a strict state
   load.
3. Requires actual training GSD coverage to span `required_gsd_coverage()`
   unless `allow_partial_gsd`; `partial_gsd` records the gap regardless.
4. Traces `model(image, gsd)` at `(1, 3, 193, 258)` with zeros GSD, opset 17,
   `dynamo=False`, dynamic batch always, and dynamic spatial dims when
   `dynamic_spatial` (including segmentor logits).
5. Validates the emitted graph metadata (unique `image`/`gsd` names, one
   shared dynamic batch symbol, `tensor(float)` dtypes, shapes) with onnx,
   computes the artifact SHA-256, writes the sidecar to a sibling temporary,
   then publishes each file with `os.link`. A failed sidecar publish removes
   the artifact link only while it is still the same file we created.

## Errors and faults

`Err` on unreadable checkpoints, missing keys, contract/coverage violations,
SDK import failures, or overwrite attempts. Nothing is promoted to active
deployment paths.

## Inputs and outputs

`export(cfg)` takes a frozen `ExportConfig` and returns `Result[Path, str]`; the value is the artifact path.

## Messages

None.

## Configuration

`ExportConfig` fields: `checkpoint_path`, `output_path`, `opset` (17), `dynamic_spatial` (true), `allow_partial_gsd` (false).

## Constraints

- Only `pactnet`/`dilatenet` conditioned checkpoints.
- No overwrite; exclusive `os.link` publish from private sibling temporaries.
- onnx and torch import lazily inside the call.

## Related documents

- [tools.ml_models.export](../export.md)
- [tools.ml_models.export.manifest](manifest.md)
- [tools.ml_models.export.contract](contract.md)
