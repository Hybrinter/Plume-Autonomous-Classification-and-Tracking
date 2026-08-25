# flight.libs.messages

**Source:** `packages/flight/src/flight/libs/messages/`
**Kind:** package

## Purpose

The messages package defines frozen dataclasses for all inter-subsystem bus traffic. Every
subsystem app publishes and subscribes using these types.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`messages`](messages/messages.md) | module | Message dataclasses, `BlobMeta`, and `utc_now_iso` |

## Package interface

`flight.libs.messages` re-exports:

| Name | Kind |
| --- | --- |
| `SCHEMA_VERSION` | constant |
| `BlobMeta` | class |
| `CommandAckMsg`, `CommandMsg`, `DownlinkItemMsg`, `FaultEventMsg` | class |
| `GimbalCommandMsg`, `HeartbeatMsg`, `InferenceResultMsg` | class |
| `LaunchLockStateMsg`, `LinkStateMsg`, `ModeChangeMsg` | class |
| `ModelDeployStateMsg`, `ModelStagedMsg`, `ProcessedFrameMsg` | class |
| `ProductRefMsg`, `RoutedCommandMsg`, `SafetyStateMsg` | class |
| `StorageWriteMsg`, `TelemetryEventMsg`, `UploadChunkMsg` | class |
| `utc_now_iso` | function |

## Interactions

Subsystem apps publish message instances on the `MessageBus`. Subscribers receive messages by
exact type match. Large numpy arrays are typed `object` in message fields. Compact records ride
the bus; large artifacts use direct storage access or inline bytes in `DownlinkItemMsg`.

## Constraints

- Every message is `@dataclass(frozen=True)`.
- `msg_type: MessageType` is the first field. `timestamp_utc: str` is the second field.
- `timestamp_utc` uses `YYYY-MM-DDTHH:MM:SS.mmmZ` format with a trailing `Z`.
- `schema_version` defaults to `SCHEMA_VERSION` (currently `1`) on every message.
- Raw sensor frames never appear as bus messages. Use `MosaicFrame` from `flight.libs.types`.
- Treat received messages as immutable. The bus delivers the same object reference to every
  subscriber.

## Related documents

- [`flight.libs`](../libs.md)
- [`flight.libs.messages.messages`](messages/messages.md)
- [`flight.libs.bus`](../bus.md)
