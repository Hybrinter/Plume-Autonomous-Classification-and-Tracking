# flight.core.model_deploy

**Source:** `packages/flight/src/flight/core/model_deploy.py`
**Kind:** module

## Purpose

The model deploy service validates staged upload artifacts and activates them on
`ACTIVATE_MODEL` with automatic rollback on failure.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SUBSYSTEM` | constant | Subsystem name `"model_deploy"` |
| `StagedModel` | class | Validated staged artifact metadata |
| `DeployState` | class | Mutable active, rollback, and staged bookkeeping |
| `ParsedManifest` | class | Parsed JSON manifest fields |
| `parse_manifest` | function | Parse manifest bytes to `ParsedManifest` or `None` |
| `contract_ok` | function | Compare manifest I/O shapes to expected shapes |
| `ModelDeployService` | class | Deploy service with `from_config`, `tick`, and `run` |

## Inputs and outputs

**`parse_manifest(blob) -> ParsedManifest | None`**

- Input: JSON manifest bytes.
- Output: parsed manifest, or `None` on malformed input.

**`contract_ok(input_shape, output_shape, expected_input, expected_output) -> bool`**

- Inputs: manifest shapes and expected inference contract shapes.
- Output: `True` when shapes match exactly.

**`ModelDeployService.from_config(cfg, bus, clock, storage_reader) -> ModelDeployService`**

- Inputs: `PactConfig`, bus, clock, `StorageReader`.
- Output: service in `ACTIVE` state with factory model version.

**`ModelDeployService.tick() -> None`**

- Processes `ModelStagedMsg` and routed `ACTIVATE_MODEL` commands.

**`ModelDeployService.run(stop_event) -> None`**

- Input: shutdown `threading.Event`.
- Runs the deploy loop with heartbeats until stop.

## Behavior

1. On `ModelStagedMsg`, read artifact bytes via `storage_reader`.
2. Verify SHA-256 digest against the announced hash.
3. Parse the JSON manifest with `parse_manifest`.
4. On success, set state to `STAGED` and publish `ModelDeployStateMsg`.
5. On read, digest, or parse failure, publish `FaultEventMsg(MODEL_CORRUPT)`.
6. On routed `ACTIVATE_MODEL`, require a staged model.
7. Compare manifest I/O shapes to the contract from inference config:
   input `(1, len(input_bands), H, W)`, output `(1, 1, H, W)`.
8. On contract match, move staged version to active, retain previous as rollback, clear
   staged, set state to `ACTIVE`, publish state, and ack accepted.
9. On contract mismatch, clear staged, set state to `ROLLBACK_AVAILABLE`, publish
   `MODEL_CORRUPT` fault, publish state, and ack rejected.
10. `run` calls `tick` each loop and emits `HeartbeatMsg` every
    `fault.watchdog_interval_s`.

## Errors and faults

Publishes `FaultEventMsg` with `FaultCode.MODEL_CORRUPT` for unreadable staged artifacts,
digest mismatch, malformed manifest, or failed activation sanity check.

Publishes `CommandAckMsg` with `FaultCode.MODEL_CORRUPT` on rejected activation.

## Messages

**Subscribes:** `ModelStagedMsg`, `RoutedCommandMsg` (target `model_deploy`,
command `ACTIVATE_MODEL`).

**Publishes:** `ModelDeployStateMsg`, `FaultEventMsg(MODEL_CORRUPT)`, `CommandAckMsg`,
`HeartbeatMsg`.

## Configuration

| Field | Source |
| --- | --- |
| `inference.input_bands` | Input channel count for contract check |
| `inference.input_height_px` | Input height for contract check |
| `inference.input_width_px` | Input width for contract check |
| `fault.watchdog_interval_s` | Heartbeat interval |

## Constraints

- Activation validates manifest I/O shapes. It does not load onnxruntime in this module.
- Failed activation keeps the previous active model and sets `ROLLBACK_AVAILABLE`.
- Initial active version is `"factory"`.
- The service emits heartbeats like other persistent-loop core services.

## Related documents

- [`flight.core`](../core.md)
- [`flight.core.storage`](storage.md)
- [`flight.core.composition`](composition.md)
