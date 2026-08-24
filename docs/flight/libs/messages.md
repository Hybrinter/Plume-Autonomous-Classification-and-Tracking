# flight.libs.messages

**Source:** `packages/flight/src/flight/libs/messages`
**Kind:** package

## Purpose

This package defines frozen dataclasses for all inter-subsystem bus messages. Every message
carries `msg_type`, `timestamp_utc`, and a trailing `schema_version`.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`messages`](flight/libs/messages/messages.md) | module | Message types, `BlobMeta`, `utc_now_iso`, `SCHEMA_VERSION` |

## Package interface

Re-exports from `flight.libs.messages.messages`:

- `SCHEMA_VERSION`
- `BlobMeta`
- `CommandAckMsg`
- `CommandMsg`
- `DownlinkItemMsg`
- `FaultEventMsg`
- `GimbalCommandMsg`
- `HeartbeatMsg`
- `InferenceResultMsg`
- `LaunchLockStateMsg`
- `LinkStateMsg`
- `ModeChangeMsg`
- `ModelDeployStateMsg`
- `ModelStagedMsg`
- `ProcessedFrameMsg`
- `ProductRefMsg`
- `RoutedCommandMsg`
- `SafetyStateMsg`
- `StorageWriteMsg`
- `TelemetryEventMsg`
- `UploadChunkMsg`
- `utc_now_iso`

## Interactions

Subsystem apps publish and subscribe to these types on `flight.libs.bus.MessageBus`. Large
numpy arrays are typed `object` in message fields. Raw mosaic frames are not bus messages;
they use `MosaicFrame` from `flight.libs.types`.

## Constraints

- Every message is `@dataclass(frozen=True)`.
- `msg_type` is the first field. `timestamp_utc` is the second field where present.
- `SCHEMA_VERSION` is `1` on every message dataclass default.
- Producers must validate array shape and dtype before publishing.

## Related documents

- [`flight.libs`](flight/libs.md)
- [`flight.libs.messages.messages`](flight/libs/messages/messages.md)
- [`flight.libs.bus`](flight/libs/bus.md)
- [`flight.libs.types`](flight/libs/types.md)
