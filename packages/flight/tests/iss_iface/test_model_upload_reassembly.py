"""Chunk-reassembly core tests: ordering, duplicates, CRC, header consistency."""

import zlib

from flight.iss_iface.upload import ModelUploadState, add_chunk
from flight.libs.types import FaultCode


def _crc(blob: bytes) -> int:
    """CRC-32 of the complete artifact."""
    return zlib.crc32(blob) & 0xFFFFFFFF


def test_single_chunk_completes() -> None:
    """A one-chunk upload reassembles immediately when the CRC matches."""
    blob = b"hello"
    state = ModelUploadState()
    result = add_chunk(state, 0, 1, blob, _crc(blob))
    assert result.complete == blob
    assert result.fault is None


def test_multi_chunk_out_of_order_completes() -> None:
    """Chunks arriving out of order still reassemble in index order."""
    blob = b"abcdef"
    crc = _crc(blob)
    state = ModelUploadState()
    assert add_chunk(state, 1, 2, b"def", crc).complete is None  # in progress
    result = add_chunk(state, 0, 2, b"abc", crc)
    assert result.complete == blob


def test_partial_upload_in_progress() -> None:
    """An incomplete upload reports progress and no completion."""
    state = ModelUploadState()
    result = add_chunk(state, 0, 3, b"a", _crc(b"abc"))
    assert result.complete is None
    assert result.fault is None


def test_bad_index_rejected() -> None:
    """An out-of-range chunk index is rejected as COMMAND_INVALID."""
    result = add_chunk(ModelUploadState(), 5, 2, b"x", 0)
    assert result.fault is FaultCode.COMMAND_INVALID


def test_inconsistent_header_rejected() -> None:
    """A chunk whose total/CRC disagrees with the first chunk is rejected."""
    state = ModelUploadState()
    add_chunk(state, 0, 2, b"ab", 123)
    result = add_chunk(state, 1, 3, b"cd", 123)  # total changed
    assert result.fault is FaultCode.COMMAND_INVALID


def test_crc_mismatch_rejected() -> None:
    """A reassembled artifact whose CRC does not match the declared value is MODEL_CORRUPT."""
    state = ModelUploadState()
    result = add_chunk(state, 0, 1, b"hello", 0xDEADBEEF)  # wrong CRC
    assert result.complete is None
    assert result.fault is FaultCode.MODEL_CORRUPT
