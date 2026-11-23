# flight.core.model_deploy

**Source:** `packages/flight/src/flight/core/model_deploy.py`
**Kind:** module

## Purpose

The model deploy service validates a staged classifier and segmentor pair bundle
and activates the pair on `ACTIVATE_MODEL`. A failed activation sanity check
keeps the previous pair active and makes rollback available.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ArtifactContract` | class | Conditioned graph input and output metadata |
| `StagedModel` | class | Validated staged pair and preprocessing metadata |
| `DeployState` | class | Active, rollback, and staged bookkeeping |
| `ParsedManifest` | class | Parsed pair-manifest fields |
| `parse_manifest` | function | Strictly parses pair-manifest JSON bytes |
| `contract_ok` | function | Compares declared image and output shapes |
| `ModelDeployService` | class | Deploy service with `from_config`, `tick`, and `run` |

## Inputs and outputs

`parse_manifest(blob)` accepts the export pair-manifest JSON format and returns
`ParsedManifest` or `None`. It rejects legacy single-input graphs, missing GSD
shapes, wrong names or dtypes, malformed dimensions, and inconsistent pair
metadata.

`ModelDeployService.from_config(cfg, bus, clock, storage_reader)` returns a
service initialized with the factory pair. `tick()` processes staged uploads
and activation commands. `run(stop_event)` emits periodic heartbeats until stop.

## Behavior

1. On `ModelStagedMsg`, read the bundle bytes and verify the announced SHA-256.
2. Parse both conditioned graph entries and stage only structurally valid pairs.
3. On `ACTIVATE_MODEL`, require a staged pair and match its grid, frame size,
   tile size, GSD reference, bands, normalization, and conditioning metadata to
   `InferenceConfig`.
4. Require each graph to use dynamic batch, exact tile spatial dimensions,
   image `(None, C, tileH, tileW)`, GSD `(None, 2)`, and one float32 output:
   classifier `(None, 1)` or segmentor `(None, 1, tileH, tileW)`.
5. On success, make the staged pair active and retain the previous version as
   rollback. On failure, keep the previous pair active and publish the rollback
   state, fault, and rejected command acknowledgement.

## Errors and faults

Publishes `FaultEventMsg(MODEL_CORRUPT)` for unreadable staged artifacts,
digest mismatch, malformed manifests, or failed activation checks.

## Messages

**Subscribes:** `ModelStagedMsg`, `RoutedCommandMsg` (target `model_deploy`,
command `ACTIVATE_MODEL`).

**Publishes:** `ModelDeployStateMsg`, `FaultEventMsg`, `CommandAckMsg`, and
`HeartbeatMsg`.

## Configuration

Uses `InferenceConfig.input_bands`, frame height and width, `tile_rows`,
`tile_cols`, and `gsd_reference_m`. Uses `FaultConfig.watchdog_interval_s` for
heartbeats.

## Constraints

- Activation checks manifest metadata and graph signatures without importing
  onnxruntime in this module.
- The classifier and segmentor are staged and activated as one pair.
- Initial active version is `factory`.

## Related documents

- [`flight.core`](../core.md)
- [`flight.core.storage`](storage.md)
- [`flight.payload.inference.contract`](../payload/inference/contract.md)
