# ADR-001: Inference Process Isolation via multiprocessing.Process

**Status:** Accepted
**Date:** 2026-04-03
**Req IDs:** REQ-AIML-COMP-002, REQ-AIML-COMP-001

## Context

The U-Net/ResNet-34 segmentation model runs on an Nvidia Jetson Xavier GPU. Python's Global
Interpreter Lock (GIL) prevents true CPU parallelism in threads, but more critically, PyTorch
CUDA operations within a thread share the GIL context with all other threads in the process.
If inference runs in the same process as storage, telemetry, or comms, a GIL contention event
or GPU memory pressure spike in one subsystem can stall inference, violating the latency budget
required to maintain gimbal tracking.

REQ-AIML-COMP-002 explicitly mandates that the inference subsystem be isolated from storage,
telemetry, and comms tasks.

## Decision

The inference subsystem runs in a dedicated `multiprocessing.Process`. This provides:
- True OS-level process isolation: the inference process has its own GIL, its own GPU context
  reservation, and its own heap.
- Fault containment: if inference crashes (NaN output, OOM, etc.), the fault process detects
  it via missed heartbeat without taking down storage or comms.
- Clean separation: inference communicates with all other subsystems exclusively via typed
  `multiprocessing.Queue` instances carrying frozen dataclass messages.

Note: preprocessing runs inside the inference process (same process, function call — see
`preprocessing/adr/ADR-001`). This keeps the hot path tight and avoids queue overhead between
preprocessing and inference.

## Consequences

### Positive
- REQ-AIML-COMP-002 is satisfied by construction.
- GPU memory can be allocated and pinned by the inference process at startup, with no risk of
  fragmentation from other subsystems.
- Model reload/rollback (`activate_staged_model`, `rollback_model`) can be triggered by sending
  a `ModeChangeMsg` to the inference process without touching any other subsystem.
- Process crash is detectable via the watchdog heartbeat mechanism.

### Negative / Trade-offs
- `multiprocessing.Queue` serializes messages via `pickle`. `np.ndarray` fields in
  `RawFrameMsg` and `ProcessedFrameMsg` are pickled on enqueue. For large frames, this adds
  serialization overhead on the hot path. Mitigation: use `multiprocessing.Queue(maxsize=8)`
  to apply backpressure and prevent unbounded memory growth.
- The inference process cannot share CUDA memory directly with the imaging process. All frame
  data crosses a queue boundary. This is acceptable given the 500 ms latency budget (TBD).
- Spawning a new process increases startup time. `main.py` must account for process warmup
  before the first inference result is expected.
