# ADR-001: threading.Thread Concurrency and SHA-256 File Integrity

**Status:** Accepted
**Date:** 2026-04-03
**Req IDs:** REQ-IMAG-HIGH-003, GOAL-003, GOAL-004

## Context

The storage subsystem receives `StorageWriteMsg` from the inference process and must persist
three files per frame: raw bands (`.npy`), processed tensor (`.npy`), and metadata (`.json`).
It then appends a `StorageRecord` to a rolling manifest file.

Storage is entirely I/O-bound (disk writes). The GIL is released during `os.write()` system
calls, so `threading.Thread` provides adequate parallelism. A separate `multiprocessing.Process`
would add overhead without benefit.

Data integrity is critical: the HSG-AIML dataset will be used for ground-based ML retraining.
A corrupted frame silently included in the training set could degrade model performance. The
system must detect write errors at write time, not at downlink time weeks later.

Two checksum options were considered:
- **MD5** — fast, but cryptographically broken and collisions are possible.
- **SHA-256** — slower but collision-resistant. Acceptable for the write frequency expected
  (limited by inference throughput, not storage speed). Configurable via
  `StorageConfig.checksum_algorithm` to allow future changes without code modification.

## Decision

1. **Concurrency: `threading.Thread`** — storage runs as a background thread in a dedicated
   storage process. The process entry point starts the thread and joins it on shutdown.
   Inter-process communication uses `multiprocessing.Queue[StorageWriteMsg]` (from inference
   process) for the cross-process boundary, and the thread reads from this queue.

2. **SHA-256 checksums** — computed after every file write using `hashlib.sha256()`. The
   computed digest is compared against a re-read of the file to verify write integrity.
   Checksum and file paths are stored in `StorageRecord` and appended to the manifest.

3. **Directory structure** — `{data_root}/{YYYY-MM-DD}/{frame_id:08d}/` — one directory per
   frame, named by zero-padded frame ID, grouped by UTC date. This structure is
   self-describing and survives partial downlinks (each frame directory is atomic).

## Consequences

### Positive
- SHA-256 detects bit-flip errors, truncated writes, and filesystem corruption at write time.
- `StorageRecord` in the manifest provides a complete audit trail for every stored frame,
  enabling `verify_manifest()` to re-check all files on demand.
- Per-frame directory structure makes partial dataset downloads safe: each frame is
  independently verifiable.
- `StorageConfig.checksum_algorithm` makes the hash choice configurable without code changes.

### Negative / Trade-offs
- SHA-256 requires reading each file back after writing to verify. This doubles the I/O per
  frame. Acceptable given that storage is not on the critical latency path (inference → controller
  latency is independent of storage write speed).
- `threading.Thread` shares the GIL with any other threads in the storage process. If a
  blocking Python operation stalls the thread (e.g., a slow manifest append), write queue
  depth will grow. Mitigation: `multiprocessing.Queue(maxsize=32)` applies backpressure.
- The manifest is append-only. Concurrent writes from multiple threads would corrupt it.
  Mitigation: a single storage thread owns all manifest writes — no locking needed.
