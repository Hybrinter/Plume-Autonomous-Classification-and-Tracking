# `iss_iface` Subsystem Context

Non-obvious context not derivable from the individual files in this package.

## Purpose / Why It Exists

- Replaces legacy RF comms. The station (ISS) now owns the RF/downlink path, so this
  subsystem does **not** modulate, schedule, or encode anything for the air — it is the
  payload's seam onto a station-owned link.

## Defining Design Decision

- Pure transport bridge, zero command interpretation. The whole subsystem is two pumps:
  - `pump_uplink`: `StationLink.receive_command()` -> `bus.publish(CommandMsg)` verbatim.
  - `pump_downlink`: drain `DownlinkItemMsg` from the bus -> `StationLink.send_downlink()`.
  The core/target apps are the ones that act on the published `CommandMsg`; this app never
  inspects, validates, routes, or transforms a command. Treat any urge to add command
  logic here as a layering violation.

## Invariants / Gotchas

- The exact ISS data-interface wire protocol is a **deliberately deferred** decision,
  hidden entirely behind the `StationLink` Protocol (`hal/interfaces/station.py`).
  `RealStationLink` is an inert stub: `receive_command()` always returns `Ok(None)`,
  `send_downlink()` accepts and drops. So with the real driver this subsystem is a no-op —
  CI/SIL exercises it through `SimStationLink`. Do not mistake the stub for broken wiring.
- Uplink vs downlink error asymmetry: an uplink `Err` **stops the drain early** (breaks the
  loop) to preserve command ordering; a downlink `Err` only emits a fault and continues
  draining the rest. Both surface failures as `FaultEventMsg` on the bus rather than raising.
- The `run()` loop reuses `fault.watchdog_interval_s` for both tick cadence and heartbeat
  cadence purely for simplicity. A production link would poll faster — the shared interval
  is not a meaningful coupling.

## Explicitly Out of Scope

- CCSDS framing, TDRSS, and contact/comm-window scheduling: those belong to the
  station's RF path, not here.
- Model-chunk upload reassembly (CRC, staging, activate/rollback) is a **future consumer**
  of this transport, not a responsibility of it. This bridge only moves opaque
  `CommandMsg`/`DownlinkItemMsg`; reassembly logic lives downstream.
