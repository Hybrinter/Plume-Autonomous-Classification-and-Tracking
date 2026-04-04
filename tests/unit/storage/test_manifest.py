"""Unit tests for pact.storage.manifest — append_to_manifest() and verify_manifest().

Satisfies: §6.2 of PACT_SW_ARCH.md — Storage subsystem unit tests.
REQ-IMAG-HIGH-003, GOAL-003, GOAL-004
"""

from __future__ import annotations

# stdlib
import json
from pathlib import Path

# third-party
import numpy as np
import pytest

# module under test
from pact.storage.manifest import append_to_manifest, verify_manifest

# pact types
from pact.types.enums import Err, FrameUsabilityTag, Ok
from pact.storage.writer import StorageRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_record(
    frame_id: int = 1,
    raw_path: str = "/data/raw.npy",
    tensor_path: str = "/data/tensor.npy",
    metadata_path: str = "/data/meta.json",
    sha256_raw: str = "abc123",
    sha256_tensor: str = "def456",
) -> StorageRecord:
    """Construct a minimal StorageRecord for manifest tests."""
    return StorageRecord(
        frame_id=frame_id,
        timestamp_utc="2026-04-03T00:00:00.000Z",
        raw_path=raw_path,
        tensor_path=tensor_path,
        metadata_path=metadata_path,
        sha256_raw=sha256_raw,
        sha256_tensor=sha256_tensor,
        usability=FrameUsabilityTag.TRAINING,
    )


# ---------------------------------------------------------------------------
# append_to_manifest tests
# ---------------------------------------------------------------------------


def test_append_creates_manifest(tmp_path: Path) -> None:
    """append_to_manifest must create the manifest file if it doesn't exist."""
    manifest_path = str(tmp_path / "manifest.jsonl")
    record = make_record(frame_id=1)

    result = append_to_manifest(record, manifest_path)
    assert isinstance(result, Ok), (
        f"Expected Ok from append_to_manifest, got {result}"
    )
    assert Path(manifest_path).exists(), "Manifest file was not created"


def test_append_writes_valid_json_line(tmp_path: Path) -> None:
    """append_to_manifest must write a valid JSON line containing frame_id."""
    manifest_path = str(tmp_path / "manifest.jsonl")
    record = make_record(frame_id=42)

    append_to_manifest(record, manifest_path)

    lines = Path(manifest_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1, f"Expected 1 JSON line, got {len(lines)}"
    data = json.loads(lines[0])
    assert data.get("frame_id") == 42, f"JSON line missing frame_id=42: {data}"


def test_append_multiple_records(tmp_path: Path) -> None:
    """Multiple append calls must produce one JSON line per record in the manifest."""
    manifest_path = str(tmp_path / "manifest.jsonl")
    for i in range(1, 4):
        append_to_manifest(make_record(frame_id=i), manifest_path)

    lines = Path(manifest_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}"


# ---------------------------------------------------------------------------
# verify_manifest tests
# ---------------------------------------------------------------------------


def _write_real_frame_and_record(tmp_path: Path, frame_id: int) -> StorageRecord:
    """Write actual numpy files and return a StorageRecord pointing to them."""
    import hashlib

    day_dir = tmp_path / "2026-04-03" / f"{frame_id:08d}"
    day_dir.mkdir(parents=True, exist_ok=True)

    raw_path = day_dir / "raw.npy"
    tensor_path = day_dir / "tensor.npy"
    metadata_path = day_dir / "meta.json"

    raw_data = np.zeros((4, 32, 32), dtype=np.float32)
    tensor_data = np.zeros((4, 32, 32), dtype=np.float32)

    np.save(str(raw_path), raw_data)
    np.save(str(tensor_path), tensor_data)
    metadata_path.write_text(json.dumps({"frame_id": frame_id}), encoding="utf-8")

    sha256_raw = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    sha256_tensor = hashlib.sha256(tensor_path.read_bytes()).hexdigest()

    return StorageRecord(
        frame_id=frame_id,
        timestamp_utc="2026-04-03T00:00:00.000Z",
        raw_path=str(raw_path),
        tensor_path=str(tensor_path),
        metadata_path=str(metadata_path),
        sha256_raw=sha256_raw,
        sha256_tensor=sha256_tensor,
        usability=FrameUsabilityTag.TRAINING,
    )


def test_verify_manifest_all_ok(tmp_path: Path) -> None:
    """verify_manifest with all files present and checksums correct returns (n, 0)."""
    manifest_path = str(tmp_path / "manifest.jsonl")
    for frame_id in range(1, 4):
        record = _write_real_frame_and_record(tmp_path, frame_id)
        append_to_manifest(record, manifest_path)

    ok_count, fail_count = verify_manifest(manifest_path, data_root=str(tmp_path))
    assert ok_count == 3, f"Expected 3 ok, got {ok_count}"
    assert fail_count == 0, f"Expected 0 failures, got {fail_count}"


def test_verify_manifest_missing_file(tmp_path: Path) -> None:
    """verify_manifest must count missing files as failures."""
    manifest_path = str(tmp_path / "manifest.jsonl")

    # Write a record pointing to a non-existent file
    bad_record = make_record(
        frame_id=999,
        raw_path=str(tmp_path / "does_not_exist.npy"),
        tensor_path=str(tmp_path / "also_missing.npy"),
    )
    append_to_manifest(bad_record, manifest_path)

    ok_count, fail_count = verify_manifest(manifest_path, data_root=str(tmp_path))
    assert fail_count >= 1, f"Expected at least 1 failure for missing file, got {fail_count}"
