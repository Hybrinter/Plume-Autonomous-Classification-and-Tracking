# Telemetry Subsystem — `pact/telemetry/`

## Purpose
Aggregate health events from all subsystems and format as CCSDS telemetry packets.

## Satisfies
- REQ-OPER-HIGH-001 — periodic system health monitoring and reporting
- REQ-COMM-HIGH-001 — downlink priority ordering (health telemetry is highest priority)

## Owns
- `DownlinkItemMsg` at `DownlinkPriority.HEALTH_TELEMETRY` — highest priority on the
  downlink queue, ensuring health data is never starved by science or imagery traffic.

## Consumes
- `TelemetryEventMsg` — from all subsystems (controller, inference, storage, comms, fault)
- `HeartbeatMsg` — sampled periodically to populate SystemHealthSnapshot fields

## Key Invariants
- Health telemetry is always enqueued at `DownlinkPriority.HEALTH_TELEMETRY`. This is
  enforced in `reporter.py` — no caller may override the priority.
- `SystemHealthSnapshot` is immutable (frozen dataclass). All fields are set at
  construction time; there is no incremental update mechanism.
- No subsystem-specific logic in telemetry — it aggregates and formats only. Business
  logic (e.g. fault thresholds) lives in `pact/fault/`.

## Concurrency
`threading.Thread` + `queue.Queue` — see `telemetry/adr/ADR-001`.

Rationale: telemetry formatting is I/O-bound (serialisation, queue puts). Threading
is sufficient; no CPU-heavy loops occur in this subsystem.

## Known Gaps / TODOs
- Thermal and power readings in `SystemHealthSnapshot` are placeholders (0.0). There is
  no hardware sensor interface yet; a hardware abstraction layer is needed for Phase II.
- Rolling `SystemHealthSnapshot` accumulation is implemented in `reporter.py`. The snapshot is emitted as a `DownlinkItemMsg` at `DownlinkPriority.HEALTH_TELEMETRY` on a periodic basis. The byte-level packet encoding uses `struct.pack` as a placeholder; full CCSDS space packet framing is Phase II.
