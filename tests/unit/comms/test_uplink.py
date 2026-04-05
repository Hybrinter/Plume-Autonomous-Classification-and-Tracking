"""Unit tests for pact.comms.uplink — activate_staged_model, rollback_model, and
process_uplink_chunk edge cases.

Satisfies: §6.2 of PACT_SW_ARCH.md — Comms subsystem unit tests.
REQ-AIML-HIGH-004 (chunked model upload with CRC verification),
REQ-AIML-HIGH-005 (staged deployment with rollback).
"""

from __future__ import annotations

# stdlib
from pathlib import Path
from unittest.mock import patch

# third-party
import pytest

# module under test
from pact.comms.ccsds import compute_crc32
from pact.comms.uplink import (
    ModelUploadSession,
    activate_staged_model,
    process_uplink_chunk,
    rollback_model,
)

# pact types
from pact.types.enums import Err, FaultCode, MessageType, ModelDeployState, Ok
from pact.types.messages import UploadChunkMsg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunk(
    chunk_index: int,
    total_chunks: int,
    data: bytes,
    expected_crc32: int,
) -> UploadChunkMsg:
    """Construct an UploadChunkMsg for uplink tests."""
    return UploadChunkMsg(
        msg_type=MessageType.UPLINK_CHUNK,
        timestamp_utc="2026-04-06T12:00:00.000Z",
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        data=data,
        expected_crc32=expected_crc32,
    )


def make_session(
    total_chunks: int,
    expected_crc32: int,
    staged_path: str,
) -> ModelUploadSession:
    """Construct a fresh ModelUploadSession with no chunks received."""
    return ModelUploadSession(
        total_chunks=total_chunks,
        received_chunks=frozenset(),
        expected_crc32=expected_crc32,
        staged_path=staged_path,
        deploy_state=ModelDeployState.STAGED,
    )


# ---------------------------------------------------------------------------
# activate_staged_model tests
# ---------------------------------------------------------------------------


def test_activate_staged_model_happy_path(tmp_path: Path) -> None:
    """activate_staged_model must move staged → active and copy old active → rollback.

    After activation:
    - active_path contains the staged content.
    - rollback_path contains the previous active content.
    - staged_path no longer exists (moved, not copied).
    """
    staged = tmp_path / "staged.pt"
    active = tmp_path / "active.pt"
    rollback = tmp_path / "rollback.pt"

    staged_content = b"new model weights"
    active_content = b"old model weights"
    staged.write_bytes(staged_content)
    active.write_bytes(active_content)

    result = activate_staged_model(str(staged), str(active), str(rollback))

    assert isinstance(result, Ok), f"Expected Ok, got Err({result.error})"
    assert active.read_bytes() == staged_content, (
        "active model should contain staged content after activation"
    )
    assert rollback.read_bytes() == active_content, (
        "rollback should contain the previous active content"
    )
    assert not staged.exists(), "staged file should have been moved (not copied)"


def test_activate_staged_model_first_deploy(tmp_path: Path) -> None:
    """activate_staged_model with no existing active model must succeed without creating rollback.

    On first deployment no prior active model exists, so rollback save is skipped.
    """
    staged = tmp_path / "staged.pt"
    active = tmp_path / "active.pt"
    rollback = tmp_path / "rollback.pt"

    staged_content = b"first model weights"
    staged.write_bytes(staged_content)
    # No active file exists — first deployment

    result = activate_staged_model(str(staged), str(active), str(rollback))

    assert isinstance(result, Ok), f"Expected Ok on first deploy, got Err({result.error})"
    assert active.read_bytes() == staged_content, (
        "active model should contain staged content on first deployment"
    )
    assert not rollback.exists(), (
        "rollback should not be created when there is no prior active model"
    )


def test_activate_staged_model_missing_staged_returns_err(tmp_path: Path) -> None:
    """activate_staged_model with no staged file must return Err(MODEL_CORRUPT)."""
    staged = tmp_path / "staged.pt"   # does NOT exist
    active = tmp_path / "active.pt"
    rollback = tmp_path / "rollback.pt"

    active.write_bytes(b"current active model")

    result = activate_staged_model(str(staged), str(active), str(rollback))

    assert isinstance(result, Err), (
        "Expected Err(MODEL_CORRUPT) when staged file is absent"
    )
    assert result.error == FaultCode.MODEL_CORRUPT


# ---------------------------------------------------------------------------
# rollback_model tests
# ---------------------------------------------------------------------------


def test_rollback_model_happy_path(tmp_path: Path) -> None:
    """rollback_model must copy rollback → active, restoring the previous model.

    The active model is overwritten with the rollback content.
    """
    active = tmp_path / "active.pt"
    rollback = tmp_path / "rollback.pt"

    failing_content = b"failing new model"
    good_content = b"good old model"
    active.write_bytes(failing_content)
    rollback.write_bytes(good_content)

    result = rollback_model(str(active), str(rollback))

    assert isinstance(result, Ok), f"Expected Ok for rollback, got Err({result.error})"
    assert active.read_bytes() == good_content, (
        "active model should contain rollback content after rollback"
    )


def test_rollback_model_missing_rollback_returns_err(tmp_path: Path) -> None:
    """rollback_model with no rollback file must return Err(MODEL_CORRUPT)."""
    active = tmp_path / "active.pt"
    rollback = tmp_path / "rollback.pt"   # does NOT exist

    active.write_bytes(b"current active model")

    result = rollback_model(str(active), str(rollback))

    assert isinstance(result, Err), (
        "Expected Err(MODEL_CORRUPT) when rollback file is absent"
    )
    assert result.error == FaultCode.MODEL_CORRUPT


# ---------------------------------------------------------------------------
# process_uplink_chunk edge cases
# ---------------------------------------------------------------------------


def test_out_of_order_chunks_fail_crc(tmp_path: Path) -> None:
    """Out-of-order chunk delivery causes CRC failure at completion.

    The implementation appends chunks in arrival order using file mode "ab", except
    for chunk_index == 0, which opens with "wb" (truncating any previously written data).
    If chunk 0 arrives after other chunks, it truncates the file, discarding the earlier
    bytes. The reassembled file is then shorter than expected and fails the full-file CRC.

    This test documents a known design constraint: callers must send chunks in order
    (chunk_index 0, 1, 2, …). Out-of-order delivery is not recoverable within a session.
    """
    full_payload = b"chunk_0_data" + b"chunk_1_data" + b"chunk_2_data"
    crc = compute_crc32(full_payload)
    staged = str(tmp_path / "staged.pt")

    session = make_session(total_chunks=3, expected_crc32=crc, staged_path=staged)

    # Send out of order: 2 first, then 0 (truncates), then 1
    out_of_order_chunks = [
        make_chunk(2, 3, b"chunk_2_data", crc),
        make_chunk(0, 3, b"chunk_0_data", crc),   # "wb" truncates the file here
        make_chunk(1, 3, b"chunk_1_data", crc),   # "ab" appends — all 3 "received"
    ]

    final_result = None
    for chunk in out_of_order_chunks:
        result = process_uplink_chunk(session, chunk)
        final_result = result
        if isinstance(result, Ok):
            session = result.value
        else:
            break   # MODEL_CORRUPT may arrive early

    # File content: chunk_0_data + chunk_1_data (chunk_2 was overwritten by the "wb" truncation).
    # Full-file CRC mismatch → MODEL_CORRUPT.
    assert isinstance(final_result, Err), (
        "Out-of-order delivery should produce CRC mismatch → Err(MODEL_CORRUPT)"
    )
    assert final_result.error == FaultCode.MODEL_CORRUPT


def test_process_uplink_chunk_write_failure_returns_err(tmp_path: Path) -> None:
    """process_uplink_chunk must return Err(MODEL_CORRUPT) if the file write fails.

    Uses unittest.mock.patch to simulate an OSError on the file open call.
    """
    crc = compute_crc32(b"model data")
    session = make_session(
        total_chunks=1,
        expected_crc32=crc,
        staged_path=str(tmp_path / "staged.pt"),
    )
    chunk = make_chunk(0, 1, b"model data", crc)

    with patch("builtins.open", side_effect=OSError("simulated write failure")):
        result = process_uplink_chunk(session, chunk)

    assert isinstance(result, Err), (
        "Expected Err(MODEL_CORRUPT) when file write fails, got Ok"
    )
    assert result.error == FaultCode.MODEL_CORRUPT
