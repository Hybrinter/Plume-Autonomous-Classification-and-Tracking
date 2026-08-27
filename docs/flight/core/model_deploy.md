# flight.core.model_deploy

**Source:** `packages/flight/src/flight/core/model_deploy.py`
**Kind:** module

## Purpose

The model deploy service validates a staged classifier+segmentor pair bundle and
activates the pair on `ACTIVATE_MODEL`. A failed sanity check rolls back the pair.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SUBSYSTEM` | constant | Subsystem name `"model_deploy"` |
| `ArtifactContract` | class | Input and output shapes for one network |
| `StagedModel` | class | Validated staged pair metadata |
| `DeployState` | class | Mutable active, rollback, and staged bookkeeping |
| `ParsedManifest` | class | Parsed JSON pair-manifest fields |
| `parse_manifest` | function | Parse manifest bytes to `ParsedManifest` or `None` |
| `contract_ok` | function | Compare one network's I/O shapes to expected shapes |
| `ModelDeployService` | class | Deploy service with `from_config`, `tick`, and `run` |

## Inputs and outputs

**`parse_manifest(blob) -> ParsedManifest | None`**

- Input: JSON pair-manifest bytes.
- Output: parsed manifest, or `None` on malformed input. A blob that lacks a
  classifier object or a segmentor object is malformed.

**`contract_ok(input_shape, output_shape, expected_input, expected_output) -> bool`**

- Inputs: one network's shapes and expected contract shapes.
- Output: `True` when shapes match exactly.

**`ModelDeployService.from_config(cfg, bus, clock, storage_reader) -> ModelDeployService`**

- Inputs: `PactConfig`, bus, clock, `StorageReader`.
- Output: service in `ACTIVE` state with the factory pair version.

**`ModelDeployService.tick() -> None`**

- Processes `ModelStagedMsg` and routed `ACTIVATE_MODEL` commands.

**`ModelDeployService.run(stop_event) -> None`**

- Input: shutdown `threading.Event`.
- Runs the deploy loop with heartbeats until stop.

## Behavior

1. On `ModelStagedMsg`, read bundle bytes via `storage_reader`.
2. Verify SHA-256 digest against the announced hash.
3. Parse the JSON pair manifest with `parse_manifest`.
4. On success, set state to `STAGED` and publish `ModelDeployStateMsg`.
5. On read, digest, or parse failure, publish `FaultEventMsg(MODEL_CORRUPT)`.
6. On routed `ACTIVATE_MODEL`, require a staged pair.
7. Compare both contracts to inference config:
   shared input `(1, len(input_bands), H, W)`, classifier output `(1, 1)`,
   segmentor output `(1, 1, H, W)`.
8. On both matches, move staged version to active, retain previous as rollback, clear
   staged, set state to `ACTIVE`, publish state, and ack accepted.
9. On either mismatch, clear staged, set state to `ROLLBACK_AVAILABLE`, publish
   `MODEL_CORRUPT` fault, publish state, and ack rejected.
10. `run` calls `tick` each loop and emits `HeartbeatMsg` every
    `fault.watchdog_interval_s`.

## Errors and faults

Publishes `FaultEventMsg` with `FaultCode.MODEL_CORRUPT` for unreadable staged
artifacts, digest mismatch, malformed pair manifest, or failed activation sanity
check.

Publishes `CommandAckMsg` with `FaultCode.MODEL_CORRUPT` on rejected activation.

## Messages

**Subscribes:** `ModelStagedMsg`, `RoutedCommandMsg` (target `model_deploy`,
command `ACTIVATE_MODEL`).

**Publishes:** `ModelDeployStateMsg`, `FaultEventMsg(MODEL_CORRUPT)`, `CommandAckMsg`,
`HeartbeatMsg`.

## Configuration

| Field | Source |
| --- | --- |
| `inference.input_bands` | Input channel count for both contracts |
| `inference.input_height_px` | Input height for both contracts |
| `inference.input_width_px` | Input width for both contracts |
| `fault.watchdog_interval_s` | Heartbeat interval |

## Constraints

- Activation validates both I/O contracts. It does not load onnxruntime in this module.
- Failed activation keeps the previous active pair and sets `ROLLBACK_AVAILABLE`.
- Initial active version is `"factory"`.
- The service emits heartbeats like other persistent-loop core services.
- The unit of stage and activate is the pair. The service does not accept a
  classifier-only or segmentor-only manifest.

## Related documents

- [`flight.core`](../core.md)
- [`flight.core.storage`](storage.md)
- [`flight.core.composition`](composition.md)
