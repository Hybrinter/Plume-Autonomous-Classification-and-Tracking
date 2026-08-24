# flight.hal.interfaces.storage

**Source:** `packages/flight/src/flight/hal/interfaces/storage.py`
**Kind:** module

## Purpose

This module defines the write and read faces of the core data store. `StorageWriter`
persists science products by direct call. `StorageReader` fetches stored bytes at downlink
time from a compact entry id.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `StorageWriter` | Protocol | Persist checksummed, quota-governed entries |
| `StorageReader` | Protocol | Read back stored bytes with checksum verify |

## Inputs and outputs

| Method | Inputs | Outputs |
| --- | --- | --- |
| `StorageWriter.store(item_id, data, priority)` | Human-readable id, raw bytes, downlink priority | `Result[str, FaultCode]` entry id |
| `StorageReader.read(entry_id)` | Storage entry id from `store` | `Result[bytes, FaultCode]` |

## Behavior

1. The payload app calls `StorageWriter.store` for large science products such as mask
   thumbnails.
2. The call returns a compact entry id.
3. The payload publishes the entry id on the bus.
4. At downlink time, iss_iface calls `StorageReader.read` with that id.
5. The reader verifies the stored checksum before returning bytes.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `STORAGE_FULL` | The entry cannot fit within the store quota |
| Checksum or missing-entry faults | Entry is absent or corrupted on read |

## Messages

None at the Protocol layer. Apps publish compact storage entry ids on the bus. Raw product
bytes stay off the bus.

## Configuration

None at the Protocol level. The concrete `StorageService` in `flight.core.storage` reads
quota and path settings from config at construction.

## Constraints

- Large artifacts pass by direct call, not on the message bus.
- The write and read faces are split. Each consumer sees only the face it needs.
- Protocols live in HAL interfaces. The concrete service lives in the composition root.

## Related documents

- [`flight.hal.interfaces`](interfaces.md)
- [`flight.core.storage`](core/storage.md)
- [`flight.payload`](payload.md)
- [`flight.iss_iface`](iss_iface.md)
