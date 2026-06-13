# `iss_iface` Subsystem Context

Non-obvious context not derivable from the individual files in this package.

## Purpose / Why It Exists

- Replaces legacy RF comms. The station (ISS) now owns the RF/downlink path, so this
  subsystem does **not** modulate, schedule, or encode anything for the air -- it is the
  payload's seam onto a station-owned link.

## Defining Design Decision (Phase 6A)

- **Authenticated command-ingress front door + downlink egress.** iss_iface is no longer a
  verbatim transport bridge. Every inbound CCSDS byte packet flows through a six-stage pure
  core (`ingress/pipeline.py`) before anything is published to the bus:

  1. **CCSDS decode + CRC verify** -- `flight.libs.ccsds.decode_packet` strips the 6-byte
     primary header, verifies the CRC-32 trailer over the whole frame (header + body + HMAC
     tag), and returns the raw body or `COMMAND_CRC_FAIL`.
  2. **JSON parse** -- the trailing 32-byte HMAC tag is split off first; the remaining body
     is decoded as UTF-8 JSON and `command_id`, `params`, `source`, and `seq` are extracted.
     Malformed JSON or missing fields yield `COMMAND_INVALID`.
  3. **HMAC-SHA256 authentication** -- the split tag is verified against the JSON body bytes
     using the injected `uplink_key`; mismatch yields `COMMAND_AUTH_FAIL`. When
     `require_auth=False` (test/bench mode) this stage is skipped.
  4. **Source allow-list check** -- `source` must appear in the configured `accepted_sources`
     tuple; an unlisted source yields `COMMAND_AUTH_FAIL`.
  5. **Typed dictionary validation** -- `lookup_command` resolves `command_id` to a
     `CommandSpec` from `COMMAND_DICTIONARY`; `validate_command` checks the params dict
     exactly against the spec's `ParamSpec` declarations. Unknown command IDs and wrong param
     kinds yield `COMMAND_INVALID`.
  6. **Sequence dedup** -- `seq` must be strictly greater than the last accepted seq for that
     source; replays and duplicates yield `COMMAND_SEQ_ERROR`. This is the final stage so
     that a spoofed or invalid packet cannot exhaust the sequence counter.

  Only packets that clear all six stages become `CommandMsg` on the bus (with `target` stamped
  from the dictionary, not from the ground frame). Every inbound packet -- accepted or rejected
  -- produces exactly one `CommandAckMsg`; rejections also emit a `FaultEventMsg`.

## Ingress Pipeline Is a Pure Core

- `ingress/pipeline.py` and its helpers are pure functions: no I/O, no bus, no clock reads.
  They map `(raw_bytes, key, require_auth, accepted_sources, last_seq_state)` to
  `(IngressOutcome, new_last_seq)` deterministically. All state is threaded in/out; the app
  shell (`IssIfaceApp`) owns the bus, clock, HMAC key, and the mutable `IngressState`.

## HMAC Key Is Injected, Never Loaded In-App

- The composition root (`flight.core.main`) reads the key bytes from
  `config.command_ingress.hmac_key_path` at startup and passes them through `build_apps` ->
  `IssIfaceApp.from_config(cfg, bus, clock, link, uplink_key)`. Unit and SIL tests pass key
  bytes directly -- no temp files, no config file I/O inside the app.

## AOS/LOS Gating

- `IssIfaceApp.pump_downlink` only drains the `CommandAckMsg` and `DownlinkItemMsg` queues
  when the link reports `LinkState.AOS`. During `LOS` the queues hold, so nothing is dropped.
  Each `tick()` publishes a `LinkStateMsg` on the bus so downstream consumers can observe link
  state transitions.

## Downlink Egress

- Acks and downlink items leave the payload as CCSDS TM packets (packet_type=0) with a
  CRC-32 trailer, encoded by `flight.libs.ccsds.encode_packet`. A sequential `tm_sequence`
  counter (14-bit wrap) is maintained in `IngressState` by the shell. Encoding failures
  (e.g. empty body) emit a `FaultEventMsg` and are not counted.

## Fault Codes Are Log-and-Continue (Never SAFE)

- `COMMAND_CRC_FAIL`, `COMMAND_AUTH_FAIL`, `COMMAND_SEQ_ERROR`, and `COMMAND_INVALID` are in
  the log-and-continue partition of `fault/policy.py`. A bad/spoofed/replayed command NACKs
  and is dropped; it never triggers `SAFE`. This is deliberate: an attacker must not be able
  to ground the payload by sending a malformed packet.

## Invariants / Gotchas

- Uplink vs downlink error asymmetry: an uplink `Err` **stops the drain early** (breaks the
  loop) to preserve command ordering; a downlink `Err` only emits a fault and continues
  draining the rest. Both surface failures as `FaultEventMsg` on the bus rather than raising.
- The `run()` loop reuses `fault.watchdog_interval_s` for both tick cadence and heartbeat
  cadence for simplicity. A production link would poll faster -- the shared interval is not a
  meaningful coupling.
- `CommandMsg.target` is stamped from `COMMAND_DICTIONARY`, not from the ground frame. The
  ground frame carries only `command_id`, `params`, `source`, and `seq`.

## Explicitly Out of Scope

- **Command router + hazardous ARM/EXECUTE two-step:** enforcing layered authority, the
  ARM/EXECUTE gated pair for hazardous commands, inhibit-at-actuation, and the `EXIT_SAFE`
  flow are Phase **6B** -- not implemented here.
- **Model-chunk upload reassembly** (CRC, staging, activate/rollback) is a **Phase 6D**
  consumer of this transport. iss_iface only moves bytes and compact records; chunk
  reassembly logic lives downstream.
- **Data system:** prioritized downlink manager and `StorageWriter`/`StorageReader` are Phase
  **6C**.
- TDRSS, RF scheduling, and contact/comm-window management: those belong to the station's
  RF path, never to this subsystem.
