# ops/ -- Agent Context

## Purpose

Entry point and orchestrator. Creates all queues, loads config, spawns all subsystem
processes and threads, and runs the mode FSM loop. The only place where the complete
process topology is visible in one file.

## Defining Design Decision

The inference process target is `_run_inference_process()` defined in `ops/main.py`,
not a function exported from `pact.model`. This function calls preprocessing inline
(function calls, no queue hop) before calling `InferenceEngine.run()`. Keeping this entry
point in `ops/` makes the co-location of preprocessing + inference explicit and visible
at the orchestration layer.

## Invariants

- All 10 inter-process queues are created in `main()` and passed as arguments. No
  subsystem creates its own queues.
- Config is loaded and validated *before* any process is spawned. A bad config crashes
  `main()` immediately -- no process starts with invalid parameters.
- `_run_inference_process()` is the *only* legal entry point for the inference process.
  Do not pass `pact.model.inference.InferenceEngine.run` directly to `Process(target=...)`.

## Gotchas

Process liveness monitoring detects dead processes via the `mode_queue` (`PROCESS_DIED`
fault from the fault subsystem) but does **not** restart them. A crashed subsystem
transitions the system to safe mode. There is no self-healing path in Phase I.

## Phase II Gaps

- Process restart on crash not implemented.
- No graceful drain of in-flight queue messages on shutdown (SIGTERM flushes stop events
  and joins threads with a 5 s timeout, then terminates remaining processes).
