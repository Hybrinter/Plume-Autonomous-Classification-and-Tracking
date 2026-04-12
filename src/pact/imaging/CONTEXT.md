# imaging/ -- Agent Context

## Purpose

Source subsystem -- produces `RawFrameMsg` from the camera. Does not consume from any
queue. Imaging is the head of the data pipeline.

## Defining Design Decision

`FlirBlackflyCamera` imports PySpin lazily inside `__init__()` only, never at module
level. This prevents `ImportError` from breaking the entire package on dev machines
without PySpin installed. The `AbstractCamera` Protocol enables `MockCamera` injection
in all tests without touching or importing `FlirBlackflyCamera`.

## Invariants

- `FlirBlackflyCamera` is never imported in tests. Any test that needs camera output
  must use `MockCamera`, which satisfies `AbstractCamera`.
- The heartbeat loop uses `threading.Event.wait(timeout=watchdog_interval_s)` -- not
  `time.sleep()` -- so it exits immediately when `stop_event` is set.

## Gotchas

None beyond the lazy PySpin import pattern described above.

## Phase II Gaps

- `FlirBlackflyCamera` is a stub -- real PySpin GigE Vision acquisition not implemented.
- Exposure and gain auto-tuning not implemented; values are fixed from config.
