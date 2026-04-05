# Operations Subsystem — `pact/ops/`

## Purpose
Top-level process orchestrator. Spawns all subsystem processes, loads config, manages
mode transitions.

## Satisfies
- REQ-OPER-HIGH-002 — system mode management and process lifecycle
- All subsystem REQ IDs indirectly, via process spawning and queue topology

## Owns
- `ModeChangeMsg` — receives from fault_process and applies mode transitions
- Coordinates all inter-process queues (created here, passed as arguments)

## Consumes
- `ModeChangeMsg` — from fault process (via mode_queue); applied to system mode FSM
- `UploadChunkMsg` — routed to model deployment logic in comms process

## Key Invariants
- Config is loaded and validated once at startup (load_config()) before any process is
  spawned. A bad config crashes immediately rather than propagating invalid parameters.
- No subsystem reads TOML directly. Each receives its typed config dataclass as an
  argument to its process entry point.
- All queues are created in main() and passed as arguments. No subsystem creates its
  own queues — this keeps the topology visible in one place.
- Preprocessing runs inside the inference process (not a separate process). This is a
  deliberate architectural decision (see preprocessing/adr/ADR-001) to keep the hot
  path latency tight and avoid queue overhead on RawFrameMsg → ProcessedFrameMsg.
- The inference process uses `_run_inference_process()` (defined in `ops/main.py`) as its
  `multiprocessing.Process` target, not a function from `pact.model`. Preprocessing +
  inference share the same process to avoid queue serialization overhead on the hot path
  (see `preprocessing/adr/ADR-001`).
- All 10 inter-process queues are created in `main()` and passed as constructor arguments.
  No subsystem creates its own queues.

## Concurrency
Main process (no separate concurrency primitive needed). All subsystems run as child
processes or threads spawned by main(). See ops/adr/ADR-001.

## Known Gaps / TODOs
- Process restart on crash not fully implemented. Currently main() detects dead processes
  via mode_queue (PROCESS_DIED fault) but does not attempt restart.
- SIGTERM/SIGINT handlers are implemented. On signal: all `threading.Event` and
  `multiprocessing.Event` stop events are set; threads are joined with a 5 s timeout;
  then remaining processes are terminated. No drain of in-flight queue messages on
  shutdown (graceful drain is Phase II).

## Implemented this session

### `_run_inference_process()`
Hot-path function that runs as `multiprocessing.Process`. Contains the full
preprocessing → inference loop:
1. Receive `RawFrameMsg` from `raw_frame_queue`
2. Radiometric calibration (`apply_calibration`)
3. Band selection (`select_bands`)
4. Quality flag computation (`compute_quality_flags`)
5. Construct `ProcessedFrameMsg` and call `engine.run()`
6. Route `InferenceResultMsg` → `inference_queue` (controller)
7. Route `StorageWriteMsg` → `storage_queue`
Heartbeat is sent to `heartbeat_queue` every `fault_cfg.watchdog_interval_s` seconds.

### Mode management loop (in `main()`)
- Drains `mode_queue` each iteration; applies validated `SystemMode` transitions via `transition_mode()`.
- Drains `uplink_queue`; routes `UploadChunkMsg` through `process_uplink_chunk()` → `activate_staged_model()` or `rollback_model()`.
- Monitors `mp_processes` liveness; emits `FaultCode.PROCESS_DIED` for any process that exits unexpectedly.
