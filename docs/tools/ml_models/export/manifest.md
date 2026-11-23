# tools.ml_models.export.manifest

**Source:** `packages/tools/src/tools/ml_models/export/manifest.py`
**Kind:** module

## Purpose

This module defines the schema-2 `ModelManifest` sidecar for a conditioned
ONNX artifact plus the sibling-path helpers. Serialization is JSON through
pydantic `TypeAdapter`; there is no pickle and no guessing.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ModelManifest` | pydantic dataclass | Frozen, extra-forbidden artifact contract |
| `sidecar_path` | function | `artifact.with_suffix(".json")` |
| `acceptance_path` | function | `artifact.with_suffix(".acceptance.json")` |
| `write_manifest` | function | Serialize a manifest as JSON |
| `load_manifest` | function | Parse and validate a sidecar |

## Behavior

`ModelManifest` validates the full contract on construction: schema 2, arch
family prefix matching `kind` (`pactnet`/`dilatenet`), the
`film-log-gsd-v1` conditioning and `ln_metres_over_reference_lateral_along`
encoding markers, exactly `BLUE`/`GREEN`/`RED` bands, `float32` input/output
dtypes, the `unit` normalization marker, finite positive ordered GSD ranges
and reference, 64-hex SHA-256 and
dataset hashes, exact default `tile_hw`/`grid`/`frame_hw`, and declared
shapes passing `verify_conditioned_shapes`.

## Errors and faults

- Construction and `load_manifest` raise `ValueError`/validation errors on
  any violation; `load_manifest` raises `OSError` on unreadable files.

## Inputs and outputs

`write_manifest(path, manifest)` writes JSON. `load_manifest(path)` returns a validated `ModelManifest`. `sidecar_path(artifact)` and `acceptance_path(artifact)` return sibling `Path`s.

## Messages

None.

## Configuration

None.

## Constraints

- JSON only; no pickle.
- Extra keys are forbidden and hashes must be 64 hex characters.

## Related documents

- [tools.ml_models.export](../export.md)
- [tools.ml_models.export.export](export.md)
- [tools.ml_models.export.session](session.md)
