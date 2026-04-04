# ADR-001: AbstractCamera Protocol for Hardware Abstraction

**Status:** Accepted
**Date:** 2026-04-03
**Req IDs:** REQ-AIML-IMAG-001, REQ-AIML-IMAG-002

## Context

The FLIR Blackfly S BFS-PGE-50S5M-C camera is accessed via PySpin (FLIR's proprietary SDK),
which is not available on PyPI and requires manual installation of the Spinnaker SDK. This
means:
- Development machines without Spinnaker SDK cannot run imaging code.
- Unit and integration tests cannot import `FlirBlackflyCamera` without failing at import time.
- The gimbal integration tests and e2e smoke test require frame data but have no camera hardware.

Additionally, the imaging subsystem is I/O-bound (waiting on GigE Vision frame delivery from
the camera), not CPU-bound. A `threading.Thread` is sufficient and simpler than a full
`multiprocessing.Process`.

## Decision

1. **`AbstractCamera` Protocol** — define a structural `typing.Protocol` with the camera
   interface. `FlirBlackflyCamera` and `MockCamera` both satisfy it without inheriting from
   a base class. PySpin is imported lazily inside `FlirBlackflyCamera.__init__` so that
   importing `camera.py` does not fail on machines without Spinnaker.

2. **`MockCamera`** — a pure-Python implementation that returns synthetic `RawFrameMsg` values
   from a configurable list. Used in all tests and in the e2e smoke test. Configurable to
   emit frames with synthetic blobs at specified frame indices.

3. **Concurrency: `threading.Thread`** — the imaging capture loop is I/O-bound (blocking on
   GigE frame delivery). The GIL is released during I/O waits, so `threading.Thread` provides
   adequate concurrency without the overhead of a separate process. Cross-thread communication
   uses `queue.Queue[RawFrameMsg]`.

## Consequences

### Positive
- All tests run without Spinnaker SDK. `MockCamera` is injected via the `AbstractCamera`
  protocol with no special test infrastructure.
- `FlirBlackflyCamera` is a complete stub with `# TODO` annotations; flight code can be added
  incrementally without changing the interface.
- `threading.Thread` is simpler to spawn and join than `multiprocessing.Process`, reducing
  startup overhead for the imaging subsystem.
- Stall detection (no frame in timeout window → `FaultEventMsg`) is straightforward to
  implement in the capture loop without inter-process signaling.

### Negative / Trade-offs
- If PySpin operations hold the GIL for extended periods (some C-extension callbacks do),
  other threads in the imaging process could be delayed. Mitigation: imaging runs as a
  dedicated thread; no CPU-heavy Python code shares its thread pool.
- `MockCamera` must be kept in sync with the `AbstractCamera` protocol as the interface
  evolves. A mypy `assert_type` check in `test_camera.py` should be added to enforce this.
- `FlirBlackflyCamera` is not tested. Hardware integration tests are out of scope for Phase I.
