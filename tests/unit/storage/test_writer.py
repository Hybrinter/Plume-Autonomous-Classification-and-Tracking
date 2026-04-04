"""Unit tests for pact.storage.writer — write_frame() and StorageRecord.

Satisfies: §6.2 of PACT_SW_ARCH.md — Storage subsystem unit tests.
REQ-IMAG-HIGH-003, GOAL-003, GOAL-004
"""

from __future__ import annotations

# stdlib
import hashlib
import json
from pathlib import Path

# third-party
import numpy as np
import pytest

# module under test
from pact.storage.writer import StorageRecord, write_frame

# pact types
from pact.types.enums import FrameUsabilityTag, MessageType, Ok
from pact.types.messages import BlobMeta, InferenceResultMsg, StorageWriteMsg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_storage_write_msg(frame_id: int = 1) -> StorageWriteMsg:
    """Construct a minimal StorageWriteMsg with synthetic numpy arrays."""
    blob = BlobMeta(
        blob_id=1,
        bbox=(10, 10, 50, 50),
        centroid_raw=(30.0, 30.0),
        pixel_area=100,
        mean_confidence=0.85,
        persistence_count=3,
    )
    inference = InferenceResultMsg(
        msg_type=MessageType.INFERENCE_RESULT,
        timestamp_utc="2026-04-03T00:00:00.000Z",
        frame_id=frame_id,
        mask=np.zeros((256, 256), dtype=np.float32),
        blobs=(blob,),
        model_version="test-v0",
        inference_ms=50.0,
        mode_flags=0,
    )
    return StorageWriteMsg(
        msg_type=MessageType.STORAGE_WRITE,
        timestamp_utc="2026-04-03T00:00:00.000Z",
        frame_id=frame_id,
        raw_frame=np.zeros((4, 256, 256), dtype=np.float32),
        processed_tensor=np.zeros((4, 256, 256), dtype=np.float32),
        inference_result=inference,
        usability=FrameUsabilityTag.TRAINING,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_write_frame_creates_files(tmp_path: Path) -> None:
    """write_frame must create raw .npy, tensor .npy, and metadata .json files on disk."""
    msg = make_storage_write_msg(frame_id=1)
    result = write_frame(msg, data_root=str(tmp_path))
    assert isinstance(result, Ok), (
        f"Expected Ok from write_frame, got Err({result.error if hasattr(result, 'error') else result})"
    )
    record: StorageRecord = result.value

    # All three files must exist
    assert Path(record.raw_path).exists(), f"Raw .npy file not found: {record.raw_path}"
    assert Path(record.tensor_path).exists(), f"Tensor .npy file not found: {record.tensor_path}"
    assert Path(record.metadata_path).exists(), (
        f"Metadata .json file not found: {record.metadata_path}"
    )


def test_write_frame_returns_storage_record(tmp_path: Path) -> None:
    """write_frame must return Ok(StorageRecord) with all required fields populated."""
    msg = make_storage_write_msg(frame_id=2)
    result = write_frame(msg, data_root=str(tmp_path))
    assert isinstance(result, Ok), f"Expected Ok, got {result}"
    record = result.value

    assert isinstance(record, StorageRecord)
    assert record.frame_id == 2
    assert record.timestamp_utc == msg.timestamp_utc
    assert record.usability == FrameUsabilityTag.TRAINING
    assert record.raw_path != ""
    assert record.tensor_path != ""
    assert record.metadata_path != ""
    assert record.sha256_raw != ""
    assert record.sha256_tensor != ""


def test_write_frame_checksum_correct(tmp_path: Path) -> None:
    """SHA-256 stored in StorageRecord must match the actual file content on disk."""
    msg = make_storage_write_msg(frame_id=3)
    result = write_frame(msg, data_root=str(tmp_path))
    assert isinstance(result, Ok), f"Expected Ok, got {result}"
    record = result.value

    # Verify raw file checksum
    raw_bytes = Path(record.raw_path).read_bytes()
    actual_raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    assert actual_raw_sha256 == record.sha256_raw, (
        f"SHA-256 mismatch for raw file: stored={record.sha256_raw}, "
        f"computed={actual_raw_sha256}"
    )

    # Verify tensor file checksum
    tensor_bytes = Path(record.tensor_path).read_bytes()
    actual_tensor_sha256 = hashlib.sha256(tensor_bytes).hexdigest()
    assert actual_tensor_sha256 == record.sha256_tensor, (
        f"SHA-256 mismatch for tensor file: stored={record.sha256_tensor}, "
        f"computed={actual_tensor_sha256}"
    )


def test_write_frame_metadata_json_valid(tmp_path: Path) -> None:
    """Metadata .json file must be parseable JSON and contain frame_id."""
    msg = make_storage_write_msg(frame_id=4)
    result = write_frame(msg, data_root=str(tmp_path))
    assert isinstance(result, Ok), f"Expected Ok, got {result}"
    record = result.value

    meta_text = Path(record.metadata_path).read_text(encoding="utf-8")
    meta = json.loads(meta_text)
    assert "frame_id" in meta, f"metadata JSON missing 'frame_id' key: {meta}"
    assert meta["frame_id"] == 4
