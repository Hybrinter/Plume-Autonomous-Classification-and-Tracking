# tools.ml_models.export.pair

**Source:** `packages/tools/src/tools/ml_models/export/pair.py`
**Kind:** module

## Purpose

This module gates flight promotion and writes the combined
classifier/segmentor pair manifest. `training_promotable` is a metadata-only
check; `flight_promotable` additionally requires a real exported artifact
that validates through `open_session`. Neither claims untested artifacts
accepted and neither copies to active deployment files.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `training_promotable` | function | Boolean run-directory metadata gate |
| `flight_promotable` | function | Strict gate: metadata plus validated exported graph |
| `write_pair_manifest` | function | `Result[dict, str]` pair manifest writer |

## Behavior

1. `training_promotable(run_dir, inference=None, allow_partial_gsd=False)`
   reads `summary.json` and requires `checkpoints/last.pt`, then verifies the
   model family, bands, unit normalization, GSD reference, conditioning
   marker, and required flight coverage. Training tile sizes are
   unconstrained — native-only and mixed extents are promotable. Malformed
   metadata returns `False`; it never raises.
2. `flight_promotable(run_dir, inference=None, allow_partial_gsd=False, *,
   artifact_path=None)` requires the metadata gate plus an exported artifact
   — the given path or any `*.onnx` beneath the run directory — whose sidecar
   matches the run's kind, arch, bands, reference, normalization,
   conditioning, dataset hash, and GSD bounds, and which `open_session`
   validates. Without a loadable graph it returns `False`.
3. `write_pair_manifest` loads both sidecars, requires one classifier and one
   segmentor, requires a passing `.acceptance.json` tied to each artifact
   SHA-256, validates both actual graphs via `open_session`, and requires
   matching conditioning, encoding, bands, normalization, reference,
   tile/grid/frame, and coverage altitude, and requires the default flight
   `gsd_reference_m`. Actual coverage must span `required_gsd_coverage()`
   unless `allow_partial_gsd`.
4. The pair JSON version is `SHA256(classifier_sha + segmentor_sha)[:16]`;
   common fields are grid, `frame_hw`, `tile_hw`, `gsd_reference_m`,
   `conditioning`, `gsd_encoding`, `coverage_altitude_m`, and `partial_gsd`.
   Each artifact entry carries `arch`, `sha256`, `input_names`,
   `input_types`, `output_type`, shapes, reference, markers, bands, and GSD
   bounds.

## Errors and faults

`write_pair_manifest` returns `Err` on missing or invalid sidecars, missing
or failed acceptance evidence, session validation failures, shared-field
mismatches, uncovered coverage, or an existing destination.

## Inputs and outputs

`training_promotable` and `flight_promotable` return `bool`. `write_pair_manifest(classifier_sidecar, segmentor_sidecar, dest, allow_partial_gsd=False)` returns `Result[dict, str]` and writes the pair JSON.

## Messages

None.

## Configuration

`inference` defaults to `InferenceConfig()`; its `input_bands` and `gsd_reference_m` define the flight preprocessing contract.

## Constraints

- Acceptance evidence must exist, pass, and match each artifact hash.
- Pair writes fail on an existing destination.
- Neither function copies to deployment paths.

## Related documents

- [tools.ml_models.export](../export.md)
- [tools.ml_models.export.session](session.md)
- [tools.ml_models.export.accept](accept.md)
