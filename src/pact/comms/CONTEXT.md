# comms/ -- Agent Context

## Purpose

CCSDS primary-header-only packet encoding, priority downlink queue, and safe model uplink
with CRC-32 verification and staged deployment (upload -> stage -> activate or rollback).

## Defining Design Decision

The comm window check is a UTC weekday gate (`MON-FRI`), not orbital contact prediction.
This is correct by design: ISS data dumps are constrained to weekdays by the ISS-ground
interface protocol. The schedule is fixed, not orbit-dependent. No Skyfield or ephemeris
integration is needed or correct for this use case.

## Invariants

- No blocking calls inside the asyncio event loop. All file I/O and blocking socket ops
  must use `asyncio.get_event_loop().run_in_executor()`.
- Daily byte budget is enforced inside `DownlinkQueue.dequeue()` -- callers never check it.
- CRC-32 is verified on every uplink chunk before any bytes touch disk.
  `process_uplink_chunk()` returns `Err(FaultCode.MODEL_CORRUPT)` on mismatch.

## Gotchas

CCSDS encoding in Phase I covers the primary header only (6 bytes: version, type, APID,
sequence flags, sequence count, data length -- all big-endian). Packet payload is
serialized with `pickle.dumps` (interim). Secondary headers, authentication fields, and
CRC-16/CCITT are Phase II.

## Phase II Gaps

- TDRSS modem hardware interface is a stub.
- Uplink chunk reassembly timeout not implemented -- incomplete uplinks accumulate.
- Full CCSDS space packet secondary headers and CRC-16/CCITT are Phase II.
