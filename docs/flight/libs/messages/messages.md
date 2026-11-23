# flight.libs.messages.messages

**Source:** `packages/flight/src/flight/libs/messages/messages.py`
**Kind:** module

## Purpose

The module defines frozen dataclasses for all inter-subsystem bus messages, plus shared helpers
and embedded structs.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SCHEMA_VERSION` | constant | Bus envelope schema version (`3`) |
| `utc_now_iso` | function | Current UTC ISO string with millisecond precision |
| `BlobMeta` | class | Embedded blob metadata struct |
| Message classes | class | One frozen dataclass per `MessageType` |

### Message classes

| Class | MessageType | Role |
| --- | --- | --- |
| `ProcessedFrameMsg` | `PROCESSED_FRAME` | Preprocessed tensor from payload preprocessing |
| `InferenceResultMsg` | `INFERENCE_RESULT` | Segmentation output to controller and storage |
| `GimbalCommandMsg` | `GIMBAL_COMMAND` | Telemetry record of an issued gimbal command |
| `TelemetryEventMsg` | `TELEMETRY_EVENT` | Structured telemetry event |
| `FaultEventMsg` | `FAULT_EVENT` | Fault notification to fault app |
| `HeartbeatMsg` | `HEARTBEAT` | Subsystem liveness signal |
| `CommandCorrelation` | struct | Command identity embedded in a mode request |
| `SystemModeRequestMsg` | `SYSTEM_MODE_REQUEST` | Request to the external mode authority |
| `SystemModeTransitionMsg` | `SYSTEM_MODE_TRANSITION` | Authority decision audit record |
| `SystemModeActivatedMsg` | `SYSTEM_MODE_ACTIVATED` | Authority activation; the payload boundary |
| `SystemModeSyncRequestMsg` | `SYSTEM_MODE_SYNC_REQUEST` | Boot-time activation snapshot request |
| `CommandMsg` | `COMMAND` | Ground command envelope from ingress |
| `RoutedCommandMsg` | `ROUTED_COMMAND` | Command accepted by router for target app |
| `SafetyStateMsg` | `SAFETY_STATE` | Fault-owned SAFE latch and active fault set |
| `StorageWriteMsg` | `STORAGE_WRITE` | Full frame bundle for storage writer |
| `ProductRefMsg` | `PRODUCT_REF` | Compact reference to a stored science product |
| `DownlinkItemMsg` | `DOWNLINK_ITEM` | Prioritized downlink queue item |
| `UploadChunkMsg` | `UPLINK_CHUNK` | One chunk of a model upload |
| `ModelStagedMsg` | `MODEL_STAGED` | Reassembled classifier+segmentor pair in storage |
| `ModelDeployStateMsg` | `MODEL_DEPLOY` | Model deployment lifecycle telemetry |
| `CommandAckMsg` | `COMMAND_ACK` | Ingress or execution acknowledgement |
| `LinkStateMsg` | `LINK_STATE` | Station link AOS/LOS state |

## Inputs and outputs

| Entry point | Inputs | Outputs |
| --- | --- | --- |
| `utc_now_iso()` | None | `str` timestamp `YYYY-MM-DDTHH:MM:SS.mmmZ` |
| Each message class | Field values per dataclass | Frozen message instance |

Common fields on every message:

| Field | Position | Description |
| --- | --- | --- |
| `msg_type` | first | `MessageType` discriminant |
| `timestamp_utc` | second | ISO 8601 UTC with millisecond precision |
| `schema_version` | trailing default | Defaults to `SCHEMA_VERSION` |

Frame-scoped messages also carry `frame_id: int` (uint32 monotonic counter).

## Behavior

1. Producers construct a frozen dataclass with `msg_type` and `timestamp_utc` set.
2. Producers call `MessageBus.publish(msg)`.
3. Subscribers receive messages by exact class match.
4. The local-only `ProcessedFrameMsg.tile_gsd_m` is a row-major `(tile_count, 2)` metres array
   consumed inside `PayloadApp.process_frame`; `ProcessedFrameMsg` is never published.
   Published `InferenceResultMsg` carries only compact tuples for tile logits, positive decisions,
   and tile GSD pairs.
5. `utc_now_iso()` formats UTC with a trailing `Z` suffix.
6. `BlobMeta` embeds in `InferenceResultMsg.blobs` with tracker ID, bbox, centroid, area,
   confidence, and persistence count.
7. `GimbalCommandMsg` records pose mode, elevation target, the active payload graph and
   node, the activation epoch and sequence, and reason. Tracking torque is not a command
   message. Compact pointing uses `TelemetryEventMsg`.
8. `CommandAckMsg` correlates via `(source, seq, command_id)`. On `REJECTED`, `fault_code`
   carries the reject reason.
9. `DownlinkItemMsg` carries inline `payload_bytes` or a non-empty `storage_ref` for large
   products fetched at transmission time.
10. `SafetyStateMsg` publishes the active SAFE-triggering fault set, latch flag, latch
    reason, evidence epoch, evidence sequence, and observation time each fault tick.
11. The system-mode family separates roles: `SystemModeRequestMsg` asks the authority for a
    transition, `SystemModeTransitionMsg` records the authority's decision as an audit trail
    (never behavior), `SystemModeActivatedMsg` is the accepted activation payload graphs
    consume, and `SystemModeSyncRequestMsg` requests the current snapshot at boot.

## Errors and faults

The module defines message shapes only. Producers emit `FaultEventMsg` with appropriate
`FaultCode` values when validation fails.

## Messages

### Published and subscribed by role

| Message | Typical publishers | Typical subscribers |
| --- | --- | --- |
| `ProcessedFrameMsg` | payload (internal; not bus in current pipeline) | inference path in payload |
| `InferenceResultMsg` | payload | telemetry observers, tests |
| `GimbalCommandMsg` | payload (pose commands) | downlink, logging |
| `TelemetryEventMsg` | any subsystem | telemetry reporter |
| `FaultEventMsg` | any subsystem | fault; payload (local containing-fault subscription) |
| `HeartbeatMsg` | monitored subsystems | fault watchdog |
| `SystemModeRequestMsg` | payload, fault | external mode authority |
| `SystemModeTransitionMsg` | external mode authority | storage, downlink |
| `SystemModeActivatedMsg` | external mode authority | payload, fault |
| `SystemModeSyncRequestMsg` | payload | external mode authority |
| `CommandMsg` | iss_iface | core command router |
| `RoutedCommandMsg` | core command router | target subsystem apps |
| `SafetyStateMsg` | fault | command router, payload |
| `StorageWriteMsg` | payload | core storage |
| `ProductRefMsg` | payload | downlink manager |
| `DownlinkItemMsg` | downlink manager | iss_iface |
| `UploadChunkMsg` | iss_iface | iss_iface upload handler |
| `ModelStagedMsg` | iss_iface | core model deploy |
| `ModelDeployStateMsg` | core model deploy | downlink |
| `CommandAckMsg` | iss_iface, target apps | downlink |
| `LinkStateMsg` | iss_iface | downlink manager |

Note: `ProcessedFrameMsg` exists as a typed record. Preprocessing outputs currently stay as
in-function values inside the payload app per the co-location invariant.

## Configuration

None in this module. Message producers read subsystem config for thresholds and limits.

## Constraints

- Every message is `@dataclass(frozen=True)`.
- `msg_type` is always the first field. `timestamp_utc` is always the second field.
- Timestamp format uses trailing `Z`, not `+00:00`.
- There is no bus message for raw sensor frames. Use `MosaicFrame` from `flight.libs.types`.
- Array fields typed `object` must be treated as immutable after publish.
- `ProcessedFrameMsg.tile_gsd_m` remains local to the co-located preprocessing/inference call.
- `TelemetryEventMsg.payload` holds JSON-serializable primitives only.

## Related documents

- [`flight.libs.messages`](../messages.md)
- [`flight.libs.types.enums`](../types/enums.md)
- [`flight.libs.bus.bus`](../bus/bus.md)
