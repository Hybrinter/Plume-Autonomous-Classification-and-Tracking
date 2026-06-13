# ADR 0009: ISS link transport + authenticated command ingress

**Status:** Accepted (2026-06-13)

**Implements:** spec Section 6 (link transport + command ingress, Phase 6A) of
`docs/superpowers/specs/2026-06-09-pact-flight-final-state-design.md`.

## Context

The 2026-06-06 baseline found `iss_iface` wired-but-inert in two distinct ways:

1. **No wire protocol.** `StationLink.receive_command()` always returned `Ok(None)` and
   `send_downlink()` dropped everything. No CCSDS framing, no real socket, no byte-level
   protocol -- the link was a permanently silent stub.

2. **Zero command authentication or validation.** The old `pump_uplink` blindly republished
   whatever `CommandMsg` the link handed up. An adversary who could inject a message onto the
   link could command the payload with no CRC check, no HMAC verification, no sequence
   deduplication, and no dictionary validation.

The baseline also identified the sensor-domain mismatch as the deepest gap (Phase 6A focuses
on the link; the remaining gaps are tracked in the FSW parity effort notes).

## Decision

### 1. The link transports raw CCSDS bytes; `iss_iface` owns framing and authentication

`StationLink` is a **byte-level transport Protocol** (`receive_packet`, `send_packet`,
`link_state`, `close`). The link hands raw framed bytes up to `iss_iface` and accepts raw
bytes for downlink. All framing, CRC verification, HMAC authentication, sequence deduplication,
and dictionary validation are the responsibility of `iss_iface`'s ingress pipeline -- not the
driver.

**Why:** the layered authority split places integrity and trust decisions in application code
(auditable, testable, driver-agnostic), not in hardware drivers. The driver is a dumb byte pipe;
`iss_iface` is the authenticated front door.

### 2. Wire format

**Telecommand (TC, inbound):**
```
[6-byte CCSDS primary header, packet_type=1, apid=tc_apid]
[body = JSON-bytes of {command_id, params, source, seq}]
[HMAC-SHA256 tag, 32 bytes]
[CRC-32 trailer, 4 bytes big-endian, over header + body + HMAC tag]
```

**Telemetry (TM, outbound):**
```
[6-byte CCSDS primary header, packet_type=0, apid=tm_apid]
[body = JSON-bytes]
[CRC-32 trailer, 4 bytes big-endian, over header + body]
```

The CCSDS `data_length` field covers `body + HMAC tag (TC only) + CRC` per the standard
(`data_length = len(body + tag + crc) - 1`). No HMAC on downlink (out of scope; the ground
trusts the physical link for Phase 6A).

### 3. CRC-32 + HMAC-SHA256 integrity and authentication

- **CRC-32** (`binascii.crc32(data) & 0xFFFFFFFF`, ISO-3309/zlib polynomial) covers the
  entire frame before the CRC bytes. Computed and verified by `flight.libs.ccsds`. This
  matches the legacy `src/pact/comms/ccsds.py` polynomial, so no format break with legacy
  ground tools.
- **HMAC-SHA256** covers the raw JSON body bytes (before the CRC is appended). The tag is
  32 bytes appended to the body inside the CCSDS data field. HMAC keys are 32-byte secrets
  loaded by the composition root; `iss_iface` receives the key bytes as a constructor arg.
  Rationale: HMAC-SHA256 is stdlib (`hmac`/`hashlib`), no new dependencies, and provides
  authenticated integrity at the command level.

### 4. Typed command dictionary is data, not dispatch

`COMMAND_DICTIONARY: dict[CommandId, CommandSpec]` maps each opcode to a frozen `CommandSpec`
(target subsystem, required `ParamSpec` list, `hazardous` flag). Validation is data-driven
iteration over `spec.params` -- no callable dispatch tables, no `getattr`. This satisfies the
strong-typing rule and is straightforwardly auditable.

`CommandMsg` stays the raw envelope (`str` target / `command_id`). The dictionary pins the
`target` canonically (the ground frame does not carry a target), so target apps receive the
same `CommandMsg` format unchanged.

### 5. Always-on ACK/NACK contract

Every inbound packet -- accepted or rejected -- produces exactly one `CommandAckMsg` on the
bus. `iss_iface` is the sole subscriber that encodes acks into CCSDS TM packets and sends them
downlink (AOS-gated). Only validated commands additionally become `CommandMsg`. This gives
ground operators deterministic feedback for every packet they send.

### 6. AOS/LOS gating

`link_state()` returns `LinkState.AOS` when a TCP client is connected, `LOS` otherwise.
`iss_iface` only drains its downlink/ack queues during AOS (acks and telemetry queue up during
LOS). `iss_iface` publishes a `LinkStateMsg` each tick so downstream consumers can observe
link state transitions.

### 7. New fault codes are log-and-continue (never SAFE)

`COMMAND_CRC_FAIL`, `COMMAND_AUTH_FAIL`, `COMMAND_SEQ_ERROR`, `COMMAND_INVALID` are added to
the log-and-continue partition of `fault/policy.py`. A bad or spoofed command NACKs and is
dropped; it never triggers `SAFE`. **Rationale:** a ground-recoverable vehicle must not be
SAFE'd by a single malformed or replayed packet -- that would give an attacker a denial-of-
service vector. The operator receives a REJECTED ack and can retransmit or diagnose.

### 8. HMAC key is injected from the composition root

The composition root (`flight.core.main`) reads the key bytes from
`config.command_ingress.hmac_key_path` at startup. SIL and unit tests pass key bytes directly.
`iss_iface` never reads config files. This follows the same pattern as `MosaicCalibration`
injection and keeps the app core free of I/O.

### 9. `RealStationLink` is a real TCP-in/UDP-out CCSDS link

- **TCP server** (lazy bind): the payload listens on `command_tcp_port`; a daemon thread does
  blocking `accept`/`recv` and enqueues raw bytes. `packet_length` from `flight.libs.ccsds`
  deframes the TCP byte stream into discrete CCSDS packets.
- **UDP client** (fire-and-forget): outbound TM packets are sent to `telemetry_udp_host:port`
  via a connected UDP socket. No connection state; a single `sendto` per packet.
- `link_state()` returns `AOS` once a client is connected and `LOS` after disconnect or before
  first connect.

## Deferred to later phases

- **Phase 6B -- command router + hazardous commands:** layered authority, ARM/EXECUTE two-step
  for hazardous commands, inhibit-at-actuation, `EXIT_SAFE` / manual-gimbal / lock-release.
- **Phase 6C -- data system:** core storage service, prioritized downlink manager,
  `StorageWriter`/`StorageReader` Protocols.
- **Phase 6D -- model upload:** chunked reassembly, stage/activate/rollback, `ModelDeployState`.

## Consequences

- **`iss_iface` is now the authenticated ingress front door.** Any future change to the wire
  format or authentication scheme is localized to `ingress/pipeline.py` and `libs/ccsds`.

- **The ingress pipeline is a pure core.** `process_inbound` has no bus access, no clock
  reads, and no I/O; it is trivially unit-testable and replayable from logs.

- **The real link is exercised in CI** via loopback-socket tests (`test_real_station_link.py`)
  that send and receive real CCSDS packets over localhost TCP and UDP.

- **No new third-party dependencies.** The entire link stack (`socket`, `struct`, `binascii`,
  `hmac`, `hashlib`, `json`, `threading`) is Python stdlib.

- **`StationLink` legacy API is removed (Phase 6A, Task 8).** `receive_command` and
  `send_downlink` are deleted from the Protocol and both drivers. Call sites that used the
  old API no longer compile -- there is no compat shim.

- **SIL command tests prove the full ingress path end-to-end.** The SIL closed-loop tests
  sign packets with `build_tc_packet`, pass them as `inbound_packets` to `build_sil_system`,
  and assert `CommandMsg` + `ACCEPTED` ack (valid key) or zero `CommandMsg` + `REJECTED` ack
  (wrong key).

- **Phase 6B forward-compat is preserved.** `CommandSpec.hazardous` exists (unused in 6A);
  `CommandAckMsg` is general enough for router/target acks; `CommandMsg` carries `target` for
  the future router; `CommandId` members are extensible.
