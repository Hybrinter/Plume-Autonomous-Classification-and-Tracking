"""Storage manifest — append-only JSON-lines record of every stored frame.

The manifest is a plain text file where each line is a JSON object representing one
StorageRecord.  It is append-only and owned by a single thread (run_storage_process).
No frame is considered stored until its manifest entry is flushed to disk.

Invariants:
  - Manifest is append-only. Lines are never removed or modified.
  - Each line is a self-contained JSON object parseable independently.
  - verify_manifest() re-checks every SHA-256 in the manifest against the actual files.

Satisfies: REQ-IMAG-HIGH-003.
"""

from __future__ import annotations

# stdlib
import json
import os

# internal
from pact.types.enums import FaultCode
from pact.types.enums import Ok, Err, Result  # type: ignore[attr-defined]
from pact.storage.writer import StorageRecord, _sha256_file


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def append_to_manifest(record: StorageRecord, manifest_path: str) -> Result[None, FaultCode]:
    """Append a StorageRecord as a JSON line to the manifest file.

    Opens the manifest in append mode and writes one JSON object followed by a newline.
    The file handle is flushed and closed after each append to ensure durability.
    Returns Ok(None) on success, Err(FaultCode.STORAGE_FULL) on I/O error.
    """
    entry: dict[str, object] = {
        "frame_id": record.frame_id,
        "timestamp_utc": record.timestamp_utc,
        "raw_path": record.raw_path,
        "tensor_path": record.tensor_path,
        "metadata_path": record.metadata_path,
        "sha256_raw": record.sha256_raw,
        "sha256_tensor": record.sha256_tensor,
        "usability": record.usability.value,
    }
    try:
        with open(manifest_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
            fh.flush()
        return Ok(None)
    except OSError:
        return Err(FaultCode.STORAGE_FULL)


def verify_manifest(manifest_path: str, data_root: str) -> tuple[int, int]:
    """Re-verify all files listed in the manifest by recomputing their SHA-256 digests.

    Reads each line of the manifest, locates the raw and tensor files, and compares the
    stored digest against the recomputed digest.

    Returns:
        (ok_count, failed_count) — counts of frames that passed and failed verification.

    Lines that cannot be parsed as JSON are counted as failures.
    Files that do not exist on disk are counted as failures.
    """
    ok_count = 0
    failed_count = 0

    if not os.path.isfile(manifest_path):
        return 0, 0

    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                raw_ok = _sha256_file(entry["raw_path"]) == entry["sha256_raw"]
                tensor_ok = _sha256_file(entry["tensor_path"]) == entry["sha256_tensor"]
                if raw_ok and tensor_ok:
                    ok_count += 1
                else:
                    failed_count += 1
            except (json.JSONDecodeError, KeyError, OSError):
                failed_count += 1

    return ok_count, failed_count
