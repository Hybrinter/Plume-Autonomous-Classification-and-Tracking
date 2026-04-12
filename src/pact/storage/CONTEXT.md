# storage/ -- Agent Context

## Purpose

Persists raw bands, processed tensors, and metadata per frame with SHA-256 integrity
checksums and an append-only manifest. The manifest flush is the atomic commit point.

## Defining Design Decision

Two checksum algorithms coexist intentionally: SHA-256 for on-disk file integrity (strong,
authoritative), and CRC-32 for in-flight CCSDS packet verification (fast, interim).
They serve different purposes and must not be conflated. A failed SHA-256 verify after
write triggers `STORAGE_FULL` fault -- not a dedicated checksum fault -- because the
effect is identical: the frame cannot be trusted and writes must halt.

## Invariants

- Manifest is append-only (opened in `'a'` mode), single-threaded, one JSON line per
  frame. Lines are never removed or modified. The manifest flush is the only commit point.
- No frame is "stored" until its manifest line is written and flushed.
- Directory structure: `{data_root}/{YYYY-MM-DD}/{frame_id:08d}/`.

## Gotchas

Date subdirectories are created per-run, not pre-created at startup. If the system clock
rolls over midnight during a run, a new date subdirectory will be created mid-session --
frames from the same acquisition session may be split across two date directories.

## Phase II Gaps

- No compression -- raw bands stored as uncompressed `.npy`.
- No LRU eviction -- `STORAGE_FULL` fault halts all writes permanently until restart.
- Downlink payload serialization uses `pickle.dumps` (interim).
