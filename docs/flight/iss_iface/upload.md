# flight.iss_iface.upload

**Source:** `packages/flight/src/flight/iss_iface/upload.py`
**Kind:** pure module

## Purpose

The upload module reassembles chunked model artifacts from authenticated `UPLOAD_MODEL_CHUNK`
commands. It accumulates chunks, verifies the declared CRC-32, and returns the complete bytes.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ModelUploadState` | class | Mutable reassembly buffer for one in-progress upload |
| `ChunkResult` | class | Per-chunk outcome: complete bytes, fault, or in-progress detail |
| `add_chunk` | function | Accumulates one chunk and reports progress or completion |

## Inputs and outputs

- `add_chunk(state, index, total, data, expected_crc32)` returns a `ChunkResult`.
- On completion, `ChunkResult.complete` holds the reassembled artifact bytes and the buffer resets.
- On rejection, `ChunkResult.fault` holds the fault code and the buffer resets on CRC failure.

## Behavior

1. Reject out-of-range indices or non-positive total chunk counts.
2. On the first chunk, record `total_chunks` and `expected_crc32`.
3. Reject later chunks whose header disagrees with the first chunk.
4. Store chunk bytes by index; duplicate indices overwrite (idempotent re-send).
5. When all chunks arrive, concatenate in index order.
6. Compare `zlib.crc32` of the blob to the declared CRC-32.
7. Reset the buffer and return complete bytes or a CRC mismatch fault.

## Errors and faults

| FaultCode | Trigger |
| --- | --- |
| `COMMAND_INVALID` | Bad chunk index, inconsistent header fields |
| `MODEL_CORRUPT` | Reassembled artifact CRC mismatch |

## Messages

None. The app shell stages completed bytes and publishes `ModelStagedMsg`.

## Configuration

None.

## Constraints

- Pure module with no bus, clock, or I/O access.
- The reassembly buffer lives in `IngressState.upload` inside the app shell.
- CRC comparison masks both values to 32 bits.

## Related documents

- [`flight.iss_iface`](../iss_iface.md)
- [`flight.iss_iface.app`](app.md)
- [`flight.core.model_deploy`](../core/model_deploy.md)
