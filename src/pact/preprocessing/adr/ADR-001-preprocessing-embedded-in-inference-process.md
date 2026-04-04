# ADR-001: Preprocessing Embedded in the Inference Process

**Status:** Accepted
**Date:** 2026-04-03
**Req IDs:** REQ-AIML-PREP-001, REQ-AIML-PREP-002, REQ-AIML-COMP-001

## Context

Preprocessing (band selection, radiometric calibration, quality flagging, crop/resize) must
run on every raw frame before inference. Two architectural options exist:

1. **Separate preprocessing process** — a dedicated process receives `RawFrameMsg` from
   imaging, runs preprocessing, and puts `ProcessedFrameMsg` on a queue for the inference
   process.
2. **Embedded in the inference process** — the inference process receives `RawFrameMsg` and
   calls preprocessing functions directly before running the model.

The latency budget for the inference + gimbal control loop is tight (500 ms placeholder,
pending Jetson Xavier benchmark — see `InferenceConfig.latency_budget_ms`). Every queue
crossing adds pickle serialization overhead for `np.ndarray` payloads.

## Decision

Preprocessing runs as a direct function call inside the inference process. There is no
separate preprocessing process and no `ProcessedFrameMsg` queue between preprocessing and
inference.

The `ProcessedFrameMsg` type still exists in `pact/types/messages.py` as the in-memory
representation passed from preprocessing functions to `InferenceEngine.run()`. It is not
serialized to a queue.

The queue boundary between imaging and inference carries `RawFrameMsg`. The queue boundary
between inference and the controller carries `InferenceResultMsg`.

## Consequences

### Positive
- Eliminates one queue crossing and one pickle/unpickle cycle on the hot path.
- Preprocessing and inference share the same process memory; no copy of the frame array is
  needed between them.
- Simpler process topology: `ops/main.py` spawns one fewer process.

### Negative / Trade-offs
- Preprocessing faults (e.g., NaN after radiometric calibration) must be handled within the
  inference process. If preprocessing panics, it takes down inference. Mitigation: preprocessing
  functions return `Result[..., FaultCode]` and the inference process emits a `FaultEventMsg`
  before exiting cleanly.
- Preprocessing cannot be benchmarked or tested in isolation as a running process. Unit tests
  for preprocessing modules remain straightforward (pure functions), but there is no integration
  test that exercises preprocessing as a standalone process.
- If preprocessing becomes CPU-bound enough to starve inference (e.g., due to per-band
  calibration on very large frames), this decision should be revisited and a separate
  preprocessing process introduced. Revisit trigger: preprocessing wall time > 20% of
  `latency_budget_ms` in Jetson benchmarks.
