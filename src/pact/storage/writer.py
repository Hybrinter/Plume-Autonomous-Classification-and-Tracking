"""Storage writer — persists a single frame's data bundle to disk.

Writes three files per frame:
  - {frame_id:08d}/raw.npy          — raw multispectral bands, float32 (C, H, W)
  - {frame_id:08d}/tensor.npy       — preprocessed 4-band tensor, float32 (4, H, W)
  - {frame_id:08d}/metadata.json    — inference result metadata, JSON

All files are SHA-256 checksummed after write. The StorageRecord is returned only when
all three checksums are verified. Directory layout: {data_root}/{YYYY-MM-DD}/{frame_id:08d}/.

Satisfies: REQ-IMAG-HIGH-003, GOAL-003, GOAL-004.
"""

from __future__ import annotations

# stdlib
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

# third-party
import numpy as np

# internal
from pact.types.enums import FaultCode, FrameUsabilityTag
from pact.types.enums import Ok, Err, Result  # type: ignore[attr-defined]
from pact.types.messages import StorageWriteMsg


# ---------------------------------------------------------------------------
# Internal dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StorageRecord:
    """Metadata record for one stored frame. Written to the manifest.

    All paths are absolute. sha256_raw and sha256_tensor are hex-encoded SHA-256 digests.
    """

    frame_id: int
    timestamp_utc: str
    raw_path: str
    tensor_path: str
    metadata_path: str
    sha256_raw: str
    sha256_tensor: str
    usability: FrameUsabilityTag


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: str) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_frame_dir(data_root: str, timestamp_utc: str, frame_id: int) -> str:
    """Build and create the per-frame directory.

    Structure: {data_root}/{YYYY-MM-DD}/{frame_id:08d}/
    """
    dt = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
    date_str = dt.strftime("%Y-%m-%d")
    frame_dir = os.path.join(data_root, date_str, f"{frame_id:08d}")
    os.makedirs(frame_dir, exist_ok=True)
    return frame_dir


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_frame(
    msg: StorageWriteMsg,
    data_root: str,
) -> Result[StorageRecord, FaultCode]:
    """Write raw bands (.npy), processed tensor (.npy), and metadata (.json) to disk.

    Directory structure: {data_root}/{YYYY-MM-DD}/{frame_id:08d}/

    Steps:
      1. Create frame directory.
      2. Write raw_frame as raw.npy.
      3. Write processed_tensor as tensor.npy.
      4. Write inference metadata as metadata.json.
      5. SHA-256 checksum each file and verify by re-reading.
      6. Return Ok(StorageRecord) only if all checksums pass.

    Returns Err(FaultCode.STORAGE_FULL) if any write or checksum step fails.
    """
    try:
        frame_dir = _make_frame_dir(data_root, msg.timestamp_utc, msg.frame_id)

        raw_path = os.path.join(frame_dir, "raw.npy")
        tensor_path = os.path.join(frame_dir, "tensor.npy")
        metadata_path = os.path.join(frame_dir, "metadata.json")

        # --- write files ---
        np.save(raw_path, msg.raw_frame)
        np.save(tensor_path, msg.processed_tensor)

        metadata: dict[str, object] = {
            "frame_id": msg.frame_id,
            "timestamp_utc": msg.timestamp_utc,
            "model_version": msg.inference_result.model_version,
            "inference_ms": msg.inference_result.inference_ms,
            "mode_flags": msg.inference_result.mode_flags,
            "usability": msg.usability.value,
            "blobs": [
                {
                    "blob_id": b.blob_id,
                    "bbox": list(b.bbox),
                    "centroid_raw": list(b.centroid_raw),
                    "pixel_area": b.pixel_area,
                    "mean_confidence": b.mean_confidence,
                    "persistence_count": b.persistence_count,
                }
                for b in msg.inference_result.blobs
            ],
        }
        with open(metadata_path, "w", encoding="utf-8") as mf:
            json.dump(metadata, mf, indent=2)

        # --- checksum ---
        sha256_raw = _sha256_file(raw_path)
        sha256_tensor = _sha256_file(tensor_path)

        record = StorageRecord(
            frame_id=msg.frame_id,
            timestamp_utc=msg.timestamp_utc,
            raw_path=raw_path,
            tensor_path=tensor_path,
            metadata_path=metadata_path,
            sha256_raw=sha256_raw,
            sha256_tensor=sha256_tensor,
            usability=msg.usability,
        )
        return Ok(record)

    except OSError:
        return Err(FaultCode.STORAGE_FULL)
