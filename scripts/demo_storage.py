"""Demo: write a synthetic StorageWriteMsg to a temp directory, read back, verify SHA-256.

Writes one synthetic frame to a temporary directory using write_frame(), reads the files
back from disk, and verifies the SHA-256 checksums stored in the StorageRecord.

Usage
-----
    python scripts/demo_storage.py

Satisfies: §7 of PACT_SW_ARCH.md (scripts/demo_storage.py)
This script is fully functional since write_frame() is a real implementation.
"""

from __future__ import annotations

# stdlib
import hashlib
import json
import sys
import tempfile
from pathlib import Path

# third-party
import numpy as np

# internal
from pact.storage.writer import write_frame
from pact.types.enums import FrameUsabilityTag, MessageType
from pact.types.enums import Ok  # type: ignore[attr-defined]
from pact.types.messages import BlobMeta, InferenceResultMsg, StorageWriteMsg


def make_storage_write_msg() -> StorageWriteMsg:
    """Build a synthetic StorageWriteMsg for the demo."""
    blob = BlobMeta(
        blob_id=1,
        bbox=(50, 50, 100, 100),
        centroid_raw=(75.0, 75.0),
        pixel_area=250,
        mean_confidence=0.88,
        persistence_count=4,
    )
    inference = InferenceResultMsg(
        msg_type=MessageType.INFERENCE_RESULT,
        timestamp_utc="2026-04-03T12:00:00.000Z",
        frame_id=1001,
        mask=np.zeros((256, 256), dtype=np.float32),
        blobs=(blob,),
        model_version="demo-v1",
        inference_ms=45.3,
        mode_flags=0,
    )
    rng = np.random.default_rng(seed=1001)
    return StorageWriteMsg(
        msg_type=MessageType.STORAGE_WRITE,
        timestamp_utc="2026-04-03T12:00:00.000Z",
        frame_id=1001,
        raw_frame=rng.random((4, 256, 256), dtype=np.float32),
        processed_tensor=rng.random((4, 256, 256), dtype=np.float32),
        inference_result=inference,
        usability=FrameUsabilityTag.TRAINING,
    )


def verify_file_sha256(file_path: str, expected_sha256: str) -> bool:
    """Verify a file's SHA-256 matches the expected hex digest."""
    actual = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()
    return actual == expected_sha256


def main() -> None:
    """Write frame to temp dir, read back, verify checksums."""
    print("PACT Storage Demo")
    print("=" * 60)

    tmp_dir = tempfile.mkdtemp(prefix="pact_demo_storage_")
    print(f"Temp directory: {tmp_dir}")

    msg = make_storage_write_msg()
    print(f"Writing frame_id={msg.frame_id} to storage...")

    result = write_frame(msg, data_root=tmp_dir)
    if not isinstance(result, Ok):
        print(f"ERROR: write_frame() failed: {result}")
        sys.exit(1)

    record = result.value
    print(f"\nStorageRecord created:")
    print(f"  frame_id:      {record.frame_id}")
    print(f"  timestamp_utc: {record.timestamp_utc}")
    print(f"  usability:     {record.usability.value}")
    print(f"  raw_path:      {record.raw_path}")
    print(f"  tensor_path:   {record.tensor_path}")
    print(f"  metadata_path: {record.metadata_path}")
    print(f"  sha256_raw:    {record.sha256_raw[:16]}...")
    print(f"  sha256_tensor: {record.sha256_tensor[:16]}...")

    # Verify raw file
    print(f"\nVerifying SHA-256 checksums...")
    raw_ok = verify_file_sha256(record.raw_path, record.sha256_raw)
    tensor_ok = verify_file_sha256(record.tensor_path, record.sha256_tensor)

    print(f"  Raw file SHA-256:    {'PASS' if raw_ok else 'FAIL'}")
    print(f"  Tensor file SHA-256: {'PASS' if tensor_ok else 'FAIL'}")

    # Read back metadata JSON
    meta = json.loads(Path(record.metadata_path).read_text(encoding="utf-8"))
    meta_ok = meta.get("frame_id") == record.frame_id
    print(f"  Metadata JSON:       {'PASS (frame_id matches)' if meta_ok else 'FAIL'}")

    if raw_ok and tensor_ok and meta_ok:
        print("\nAll checks PASSED. Storage write/verify pipeline is working correctly.")
    else:
        print("\nSome checks FAILED. Review storage/writer.py implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
