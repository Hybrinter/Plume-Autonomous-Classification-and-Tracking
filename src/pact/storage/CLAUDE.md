# Storage Subsystem — `pact/storage/`

## Purpose
Persist raw frames, processed tensors, and inference metadata with checksums and manifests.

## Satisfies
- REQ-IMAG-HIGH-003 — all captured frames must be stored with integrity guarantees
- GOAL-003 — on-orbit dataset accumulation for ground retraining
- GOAL-004 — downlink-ready metadata for ground-based filtering

## Owns
- `StorageRecord` (internal dataclass) — metadata record per stored frame
- Produces `DownlinkItemMsg` onto the downlink queue (science data priority)

## Consumes
- `StorageWriteMsg` — from inference process via `storage_queue`

## Key Invariants
- Every file is checksummed (SHA-256) after write and verified before the manifest entry is
  written. A write that fails checksum verification is treated as a STORAGE_FULL fault.
- Manifest is append-only (open in 'a' mode) and is owned by a single thread — no concurrent
  writers. This is enforced by the single-threaded `run_storage_process()` design.
- Directory structure is `{data_root}/{YYYY-MM-DD}/{frame_id:08d}/`.
- No frame is considered stored until its manifest entry is flushed to disk. The manifest
  flush is the commit point.

## Concurrency
`threading.Thread` + `multiprocessing.Queue` bridge — see `storage/adr/ADR-001`.

Rationale: storage writes are I/O-bound (disk writes); `threading.Thread` is sufficient and
avoids the overhead of a second OS process. A `multiprocessing.Queue` bridges from the
inference process (a separate OS process) to the storage thread.

## Known Gaps / TODOs
- No compression for raw bands (.npy is uncompressed). Compression would reduce disk use
  significantly but is deferred to Phase II.
- No eviction policy when storage is full — the process emits a STORAGE_FULL fault and halts
  further writes. An LRU eviction policy is out of scope for Phase I.
