# flight.libs.messages.messages

**Source:** `packages/flight/src/flight/libs/messages/messages.py`
**Kind:** module

## Purpose

This module defines all bus message dataclasses and shared helpers. Messages are immutable
value types routed by exact type on the message bus.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SCHEMA_VERSION` | constant | Bus envelope schema version (`1`) |
| `utc_now_iso` | function | Current UTC time as ISO 8601 with millisecond `Z` suffix |
| `BlobMeta` | dataclass | Single detected blob metadata embedded in inference results |

### Message dataclasses

| Name | `MessageType` | Key fields |
| --- | --- | --- |
| `ProcessedFrameMsg` | `PROCESSED_FRAME` | `frame_id`, `tensor`, `quality_flags`, crop metadata |
| `InferenceResultMsg` | `INFERENCE_RESULT` | `frame_id`, `mask`, `blobs`, `model_version`, crop metadata |
| `GimbalCommandMsg` | `GIMBAL_COMMAND` | `mode`, axis values, arbiter `state`, `reason` |
| `TelemetryEventMsg` | `TELEMETRY_EVENT` | `subsystem`, `event_name`, JSON-serializable `payload` |
| `FaultEventMsg` | `FAULT_EVENT` | `fault_code`, `subsystem`, `detail` |
| `HeartbeatMsg` | `HEARTBEAT` | `subsystem`, `sequence` |
| `ModeChangeMsg` | `MODE_CHANGE` | `new_mode`, `requested_by` |
| `CommandMsg` | `COMMAND` | `target`, `command_id`, `params`, `source`, `seq` |
| `RoutedCommandMsg` | `ROUTED_COMMAND` | Same envelope as `CommandMsg` after router acceptance |
| `SafetyStateMsg` | `SAFETY_STATE` | `mode`, `active_faults`, `safe_latched`, `safe_reason` |
| `StorageWriteMsg` | `STORAGE_WRITE` | `raw_frame`, `processed_tensor`, `inference_result`, `usability` |
| `ProductRefMsg` | `PRODUCT_REF` | Storage `entry_id`, downlink `priority`, `byte_len` |
| `DownlinkItemMsg` | `DOWNLINK_ITEM` | `priority`, inline or storage-backed payload |
| `UploadChunkMsg` | `UPLINK_CHUNK` | Chunk index, data bytes, expected file CRC |
| `ModelStagedMsg` | `MODEL_STAGED` | Staged artifact `entry_id`, `sha256`, `version` |
| `ModelDeployStateMsg` | `MODEL_DEPLOY` | Deploy `state`, `version`, `detail` |
| `CommandAckMsg` | `COMMAND_ACK` | `status`, echoed command ids, `fault_code`, `detail` |
| `LinkStateMsg` | `LINK_STATE` | Station link `state` |
| `LaunchLockStateMsg` | `LAUNCH_LOCK_STATE` | Launch-lock `state` |

## Inputs and outputs

- `utc_now_iso() -> str` returns `YYYY-MM-DDTHH:MM:SS.mmmZ`.
- Each message dataclass is constructed with keyword arguments for its fields.

## Behavior

1. `utc_now_iso` formats UTC wall time with three fractional digits and a `Z` suffix.
2. `BlobMeta` holds blob id, bounding box, centroid, area, mean confidence, and persistence
   count.
3. `ProcessedFrameMsg` carries a `(4, H, W)` float32 tensor and quality flags.
4. `InferenceResultMsg` carries a float32 mask, blob list, timing, and preprocess crop
   metadata for back-projection.
5. `GimbalCommandMsg` records a issued gimbal command for telemetry. Axis values mean rate
   (deg/s) in `RATE` mode and target angle (deg) in `ABSOLUTE` mode.
6. `CommandMsg` is the standard uplink command envelope. `RoutedCommandMsg` is the router
   output consumed by target apps.
7. `SafetyStateMsg` publishes fault-owned SAFE latch state each tick.
8. `StorageWriteMsg` bundles one frame's raw data, processed tensor, and inference result for
   the storage writer.
9. `ProductRefMsg` is a compact downlink reference after large artifacts are stored off-bus.
10. `DownlinkItemMsg` carries either inline `payload_bytes` or a non-empty `storage_ref`.
11. `CommandAckMsg` correlates to the originating command via `(source, seq, command_id)`.
12. On `ACCEPTED`, `fault_code` is `FaultCode.NONE`. On `REJECTED`, `fault_code` carries the
    reject reason.

## Errors and faults

None at construction. Fault codes appear as field values inside `FaultEventMsg` and
`CommandAckMsg`.

## Messages

This module defines the message types. Apps publish and subscribe to these classes on the
bus. The module does not publish or subscribe itself.

## Configuration

None.

## Constraints

- Large arrays (`tensor`, `mask`, `raw_frame`, and similar) are typed `object`, not
  `np.ndarray`.
- Every message includes `schema_version: int = SCHEMA_VERSION` as a trailing field.
- Raw sensor mosaic frames are not bus messages.

## Related documents

- [`flight.libs.messages`](flight/libs/messages.md)
- [`flight.libs.types`](flight/libs/types.md)
- [`flight.libs.bus`](flight/libs/bus.md)
