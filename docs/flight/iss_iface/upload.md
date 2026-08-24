# flight.iss_iface.upload

**Source:** `packages/flight/src/flight/iss_iface/upload.py`
**Kind:** pure module

## Purpose

This module reassembles a chunked model upload into one artifact byte string. It accumulates
chunks in a buffer, verifies the declared CRC-32, and reports completion or rejection.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ModelUploadState` | class | Mutable reassembly buffer for one in-progress upload |
| `ChunkResult` | class | Per-chunk outcome with optional complete bytes or fault |
| `add_chunk` | function | Accumulates one chunk and reports progress or completion |

## Inputs and outputs

`add_chunk(state, index, total, data, expected_crc32)` mutates `state` in place and returns a
`ChunkResult`.

When reassembly is incomplete, `complete` is `None` and `fault` is `None`.

When reassembly succeeds, `complete` holds the artifact bytes and `fault` is `None`.

When the chunk is rejected, `complete` is `None` and `fault` holds the fault code.

## Behavior

1. Reject when `total` is zero or negative, or when `index` is out of range.
2. On the first chunk, record `total` and `expected_crc32` in state.
3. Reject when a later chunk disagrees with the first chunk header.
4. Store the chunk bytes at `index`. Return an in-progress result when chunks remain.
5. When all chunks are present, concatenate in index order and verify CRC-32.
6. Reset the buffer on completion or CRC failure.

Duplicate chunk indices overwrite prior data for that index.

## Errors and faults

| Fault code | Trigger |
| --- | --- |
| `COMMAND_INVALID` | Bad chunk index or inconsistent chunk header |
| `MODEL_CORRUPT` | Reassembled artifact CRC mismatch |

## Messages

None.

## Configuration

None.

## Constraints

The module is pure. It performs no I/O, reads no clock, and touches no bus. The app shell
calls `add_chunk`, stores the completed artifact, and publishes `ModelStagedMsg`.

## Related documents

- [`flight.iss_iface`](../iss_iface.md)
- [`flight.iss_iface.app`](app.md)
