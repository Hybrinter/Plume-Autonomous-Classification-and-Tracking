"""Chunked model-upload reassembly buffer for iss_iface (no bus, no clock, no I/O).

A model artifact is uploaded as a sequence of authenticated UPLOAD_MODEL_CHUNK commands; this
module reassembles them into the complete artifact bytes. add_chunk accumulates chunks into a
ModelUploadState buffer (held by the app shell, like the ingress sequence map) and, on the final
chunk, concatenates them in index order and verifies the declared CRC-32 of the whole artifact.
The shell then stages the assembled bytes into storage and announces a ModelStagedMsg.

Out-of-range indices or an inconsistent per-chunk header (total/CRC disagreeing with the first
chunk) reject as COMMAND_INVALID; a final CRC mismatch rejects as MODEL_CORRUPT (and resets the
buffer). Duplicate chunk indices overwrite (idempotent re-send).

Contains:
  - ModelUploadState: the mutable reassembly buffer (total, expected CRC, index -> bytes).
  - ChunkResult: per-chunk outcome (assembled bytes when complete, or a fault, or in-progress).
  - add_chunk: accumulate one chunk and report progress / completion / rejection.

Satisfies: REQ-COMM-MODEL-001.
"""

from __future__ import annotations

# stdlib
import zlib
from dataclasses import dataclass, field

# internal
from flight.libs.types import FaultCode


@dataclass(slots=True)
class ModelUploadState:
    """Mutable reassembly buffer for one in-progress chunked model upload.

    Fields:
        total_chunks: Expected chunk count (0 until the first chunk sets it).
        expected_crc32: Declared CRC-32 of the complete artifact (0 until the first chunk).
        chunks: Received chunk index -> raw bytes.
    """

    total_chunks: int = 0
    expected_crc32: int = 0
    chunks: dict[int, bytes] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ChunkResult:
    """Outcome of accumulating one chunk.

    Fields:
        complete: The reassembled artifact bytes once all chunks are in and the CRC matches,
            else None.
        fault: A FaultCode if the chunk was rejected (COMMAND_INVALID / MODEL_CORRUPT), else None.
        detail: Human-readable progress/rejection context.
    """

    complete: bytes | None
    fault: FaultCode | None
    detail: str


def add_chunk(
    state: ModelUploadState, index: int, total: int, data: bytes, expected_crc32: int
) -> ChunkResult:
    """Accumulate one upload chunk into state; report progress, completion, or rejection.

    Args:
        state: The reassembly buffer (mutated in place; reset on completion or CRC failure).
        index: Zero-based chunk index.
        total: Total chunk count declared by this chunk.
        data: Raw chunk bytes.
        expected_crc32: Declared CRC-32 of the complete artifact (repeated on every chunk).

    Returns:
        A ChunkResult: complete bytes when the artifact is fully reassembled and CRC-verified,
        a fault on rejection, or an in-progress result otherwise.
    """
    if total <= 0 or index < 0 or index >= total:
        return ChunkResult(None, FaultCode.COMMAND_INVALID, f"bad chunk index {index}/{total}")
    if state.total_chunks == 0:
        state.total_chunks = total
        state.expected_crc32 = expected_crc32
    elif state.total_chunks != total or state.expected_crc32 != expected_crc32:
        return ChunkResult(None, FaultCode.COMMAND_INVALID, "inconsistent chunk header")

    state.chunks[index] = data
    if len(state.chunks) < state.total_chunks:
        return ChunkResult(None, None, f"chunk {len(state.chunks)}/{state.total_chunks}")

    blob = b"".join(state.chunks[i] for i in range(state.total_chunks))
    expected = state.expected_crc32
    state.total_chunks = 0
    state.expected_crc32 = 0
    state.chunks = {}
    if (zlib.crc32(blob) & 0xFFFFFFFF) != (expected & 0xFFFFFFFF):
        return ChunkResult(None, FaultCode.MODEL_CORRUPT, "reassembled artifact CRC mismatch")
    return ChunkResult(blob, None, "reassembly complete")
