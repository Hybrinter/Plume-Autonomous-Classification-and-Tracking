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

## Concurrency
Main process (no separate concurrency primitive needed). All subsystems run as child
processes or threads spawned by main(). See ops/adr/ADR-001.

## Known Gaps / TODOs
- Process restart on crash not fully implemented. Currently main() detects dead processes
  via mode_queue (PROCESS_DIED fault) but does not attempt restart.
- Graceful shutdown signal handling (SIGTERM/SIGINT) is a stub — processes are
  terminated by joining with a timeout then killing. Phase II will implement clean drain.
