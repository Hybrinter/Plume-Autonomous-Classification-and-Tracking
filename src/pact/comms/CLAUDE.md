# Comms Subsystem

## Purpose
CCSDS packet encoding/decoding, downlink priority queue management, uplink model staging
with CRC verification, and communication pass window scheduling.

## Satisfies
- REQ-COMM-HIGH-001 — comm window enforcement (weekdays only via TDRSS)
- REQ-COMM-HIGH-002 — daily byte budget enforcement (1 GB downlink / 100 MB uplink)
- REQ-COMM-HIGH-003 — CCSDS Space Packet Protocol encoding for all downlinked data
- GOAL-004 — safe model uplink with staged deployment and rollback capability
- GOAL-008 — priority-ordered downlink (health telemetry > science > compressed > raw)

## Owns (produces)
- `DownlinkItemMsg` — dequeues from the priority queue and transmits to TDRSS radio
- `FaultEventMsg` — emitted with FaultCode.COMM_TIMEOUT when a comm window opens but
  the radio interface does not acknowledge within the configured timeout

## Consumes
- `DownlinkItemMsg` — received from storage process (imagery) and telemetry process (health)
- `UploadChunkMsg` — received from uplink handler (ground-commanded model chunks)

## Key Invariants
- **No blocking calls inside the asyncio event loop.** All file I/O and blocking socket
  operations must be dispatched via `asyncio.get_event_loop().run_in_executor()`.
- **Daily byte budget is enforced before every dequeue.** `DownlinkQueue.dequeue()` checks
  `bytes_remaining_today()` before returning an item; callers never need to recheck.
- **Comm window checked before every send.** `is_comm_window_open()` is called at the
  start of every send attempt; items are not dequeued if the window is closed.
- **CRC verified on every uplink chunk.** `process_uplink_chunk()` returns
  Err(FaultCode.MODEL_CORRUPT) on any CRC mismatch; no partial chunks are written.
- See comms/adr/ADR-001 for the asyncio concurrency choice rationale.
- `process_uplink_chunk` is a module-level function exported from `pact.comms.uplink`. Import it at the top of any consumer module — do not use an inline `from pact.comms.uplink import process_uplink_chunk` inside a loop body.

## Concurrency
`asyncio` — the comms subsystem multiplexes many concurrent I/O waiters (radio socket,
uplink chunk reassembly timer, downlink queue drain, pass window scheduler) without
CPU-heavy work. asyncio is idiomatic for this pattern. No CPU-heavy loops exist in comms.

## Known Gaps / TODOs
- TDRSS radio hardware interface is a **stub**. Current implementation writes to a file
  or local socket. Replace with the vendor TDRSS modem API before flight integration.
- No real RF link simulation. Pass window scheduling uses UTC weekday check only; no
  orbital mechanics or Doppler correction.
- Uplink chunk reassembly timeout (session expiry) is not yet implemented.
