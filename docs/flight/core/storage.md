# flight.core.storage

**Source:** `packages/flight/src/flight/core/storage.py`
**Kind:** module

## Purpose

The storage service persists checksummed product files, enforces a byte quota, and appends
telemetry and fault records to reboot-surviving logs.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SUBSYSTEM` | constant | Subsystem name `"storage"` |
| `EntryMeta` | class | Index record for one stored entry |
| `StorageState` | class | Mutable entry index and byte counters |
| `StorageService` | class | Store, read, tick, run, and read_fault_ledger |

## Inputs and outputs

**`StorageService.from_config(cfg, bus, clock) -> StorageService`**

- Inputs: `PactConfig`, shared `MessageBus`, `Clock`.
- Output: service with empty index. No filesystem I/O at construction.

**`StorageService.store(item_id, data, priority) -> Result[str, FaultCode]`**

- Inputs: product id string, raw bytes, downlink priority.
- Output: `Ok(entry_id)` on success, or `Err(STORAGE_FULL)`.

**`StorageService.read(entry_id) -> Result[bytes, FaultCode]`**

- Input: entry id from `store`.
- Output: `Ok(data)` when checksum matches, or `Err(STORAGE_CORRUPT)`.

**`StorageService.tick() -> None`**

- Appends pending telemetry and fault messages to JSON-line logs.

**`StorageService.read_fault_ledger() -> list[dict[str, object]]`**

- Output: parsed fault ledger records oldest first, or an empty list when missing.

**`StorageService.run(stop_event) -> None`**

- Input: shutdown `threading.Event`.
- Runs the persistence loop with heartbeats until stop.

## Behavior

1. `store` rejects items larger than `max_storage_bytes`.
2. `store` evicts lowest-priority then oldest entries until the new item fits.
3. Eviction stops when only higher-priority entries remain. Further admission returns
   `STORAGE_FULL`.
4. Each stored product writes payload bytes and a SHA-256 sidecar under
   `{data_root}/products/`.
5. Entry ids use the form `{order:08d}_{item_id}`.
6. `read` verifies the sidecar checksum before returning bytes.
7. `tick` drains `TelemetryEventMsg` into `telemetry.jsonl`.
8. `tick` drains `FaultEventMsg` into `fault_ledger.jsonl`.
9. Parent directories are created lazily on first write.
10. `run` calls `tick` each loop and emits `HeartbeatMsg` every
    `fault.watchdog_interval_s`.

## Errors and faults

| Result / fault | Trigger |
| --- | --- |
| `Err(STORAGE_FULL)` | Item exceeds quota, or no evictable space remains |
| `Err(STORAGE_CORRUPT)` | Missing entry, read error, or checksum mismatch |
| `FaultEventMsg(STORAGE_FULL)` | Published alongside `Err(STORAGE_FULL)` |

## Messages

**Subscribes:** `TelemetryEventMsg`, `FaultEventMsg`.

**Publishes:** `FaultEventMsg(STORAGE_FULL)`, `HeartbeatMsg`.

## Configuration

| Field | Source |
| --- | --- |
| `storage.data_root` | Root directory for products and logs |
| `storage.max_storage_bytes` | Total byte quota |
| `storage.checksum_algorithm` | Sidecar algorithm name (SHA-256 used in code) |
| `fault.watchdog_interval_s` | Heartbeat interval |

## Constraints

- One shared storage instance serves payload, iss_iface, and model_deploy via injection.
- The service implements `StorageWriter.store` and `StorageReader.read` protocols.
- No filesystem I/O runs at construction time.
- Eviction never removes an entry with higher downlink priority than the incoming item.
- The service emits heartbeats like other persistent-loop core services.

## Related documents

- [`flight.core`](../core.md)
- [`flight.core.downlink`](downlink.md)
- [`flight.core.model_deploy`](model_deploy.md)
- [`flight.core.composition`](composition.md)
