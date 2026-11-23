# tools.ml_models.export.session

**Source:** `packages/tools/src/tools/ml_models/export/session.py`
**Kind:** module

## Purpose

This module opens a validated ONNX Runtime session for a conditioned
artifact. The artifact hash is verified before the SDK loads the file, and
the declared graph metadata must satisfy the two-input contract. There is no
one-input fallback; the legacy loader is unchanged for #104.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `Node` | dataclass | Declared `name`, `shape`, `type` of one graph value |
| `Session` | protocol | `get_inputs`, `get_outputs`, `run` |
| `open_session` | function | `Result[Session, str]` hash-first session opener |

## Behavior

1. Computes the artifact SHA-256 and compares it to `manifest.sha256` before
   constructing the runtime session.
2. Loads onnxruntime lazily with `CPUExecutionProvider`; SDK failures return
   `Err`.
3. Requires exactly `{"image", "gsd"}` inputs, one output, and
   `tensor(float)` dtypes; `input_types`/`output_type` on the manifest record
   this validated dtype.
4. Requires a dynamic batch on every I/O tensor sharing one named symbol, or
   all `None`; fixed or mixed batch dims fail.
5. Normalizes symbolic dims to `None`, requires equality with the manifest
   shapes, then applies `verify_conditioned_shapes`.

## Errors and faults

Every failure — unreadable artifact, hash mismatch, SDK error, malformed
metadata, or contract violation — returns `Err` with a reason string.

## Inputs and outputs

`open_session(artifact, manifest)` returns `Result[Session, str]`; `Session.run` takes `(output_names, feeds)` and returns a list of output arrays.

## Messages

None.

## Configuration

None. The loader always uses `CPUExecutionProvider`.

## Constraints

- Hash verification precedes SDK load.
- No one-input fallback.
- onnxruntime imports lazily.

## Related documents

- [tools.ml_models.export](../export.md)
- [tools.ml_models.export.manifest](manifest.md)
- [flight.payload.inference.contract](../../../flight/payload/inference/contract.md)
