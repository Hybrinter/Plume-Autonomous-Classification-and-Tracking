# flight.hal.interfaces.storage

**Source:** `packages/flight/src/flight/hal/interfaces/storage.py`
**Kind:** module

## Purpose

Defines the write and read faces of the core data store. The payload persists large
science products by direct call through `StorageWriter`. `iss_iface` fetches stored
bytes at downlink time through `StorageReader`. The concrete service lives in
`flight.core.storage`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `StorageWriter` | class | Runtime-checkable Protocol for persisting entries |
| `StorageReader` | class | Runtime-checkable Protocol for reading entries back |

## Inputs and outputs

`StorageWriter.store`:

- Inputs: `item_id` (str), `data` (bytes), `priority` (`DownlinkPriority`)
- Output: `Result[str, FaultCode]` — the storage entry id on success

`StorageReader.read`:

- Input: `entry_id` (str)
- Output: `Result[bytes, FaultCode]`

## Behavior

1. `store` persists bytes under a human-readable `item_id`. It returns a storage entry
   id for later reads.
2. `read` fetches the bytes for an entry id and verifies the stored checksum.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `FaultCode.STORAGE_FULL` | The entry cannot fit within the storage quota |
| `FaultCode.STORAGE_CORRUPT` | The entry is missing or its checksum does not match |

## Messages

None. Large artifacts bypass the bus.

## Configuration

None at the Protocol level. The concrete service reads `StorageConfig` at construction.

## Constraints

- Two separate Protocols isolate write and read consumers.
- `store` retains the downlink priority for quota eviction ordering.

## Related documents

- [`flight.hal.interfaces`](../interfaces.md)
