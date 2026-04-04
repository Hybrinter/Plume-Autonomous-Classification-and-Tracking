# Imaging Subsystem

## Purpose
Interface with the FLIR Blackfly S BFS-PGE-50S5M-C camera over GigE Vision and manage
frame capture, stall detection, and delivery of raw frames to the preprocessing pipeline.

## Satisfies
- REQ-AIML-IMAG-001 — raw multi-spectral frame acquisition at required cadence
- REQ-AIML-IMAG-002 — per-frame exposure and gain metadata recorded with each frame

## Owns (produces)
- `RawFrameMsg` — one per captured frame, placed on raw_frame_queue for preprocessing
- `FaultEventMsg` — emitted with FaultCode.CAMERA_STALL when no frame is received within
  the configured stall timeout
- `HeartbeatMsg` — sent to the fault watchdog on each watchdog interval

## Consumes
Nothing. Imaging is a **source subsystem** — it does not read from any queue. All
inter-subsystem communication is outbound only.

## Key Invariants
- `FlirBlackflyCamera` is **never imported in tests**. Any test that needs a camera must
  use `MockCamera`, which satisfies the `AbstractCamera` Protocol.
- `MockCamera` satisfies `AbstractCamera` for all test uses and is configurable with a
  list of synthetic frames to return in sequence.
- PySpin is imported **lazily** inside `FlirBlackflyCamera.__init__()` only. It is never
  imported at module level. This prevents an ImportError from breaking the entire package
  when PySpin is not installed (e.g., on a dev laptop).
- The imaging process uses `threading.Thread` (not `multiprocessing.Process`) because
  camera capture is I/O-bound. See imaging/adr/ADR-001.

## Concurrency
`threading.Thread` + `queue.Queue` — frame capture is I/O-bound (GigE Vision DMA transfer).
The GIL is not a bottleneck here, and thread-based capture avoids pickling overhead for
large numpy frame arrays across process boundaries.

## Known Gaps / TODOs
- `FlirBlackflyCamera` is a **stub**. PySpin GigE Vision integration not yet complete.
  Replace the stub body with real PySpin acquisition calls before hardware integration.
- No hardware integration tests exist. The CI pipeline uses MockCamera exclusively.
- Exposure and gain auto-tuning logic is not yet implemented (fixed values from config).
