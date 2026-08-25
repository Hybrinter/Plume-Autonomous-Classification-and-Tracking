# gse.station

**Source:** `packages/gse/src/gse/station.py`
**Kind:** module

## Purpose

`StationEmulator` is the test-side counterpart to flight `RealStationLink`. It sends signed
telecommands over TCP and receives telemetry datagrams over UDP.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `StationEmulator` | class | Connect, send_command, poll_downlink, close |

## Inputs and outputs

**`StationEmulator.__init__(tcp_host, tcp_port, udp_host, udp_port, key, tc_apid)`**

- Records endpoints and HMAC key. Opens no sockets until `connect`.

**`connect() -> None`**

- Opens UDP receiver and TCP client to the flight link.
- Raises `RuntimeError` if already connected. Raises `OSError` on bind or connect failure.

**`send_command(command_id, params, source, seq) -> None`**

- Builds a packet with `build_tc_packet` and sends it on the TCP connection.
- Raises `RuntimeError` before connect. Propagates `ValueError` from the packet builder.

**`poll_downlink(timeout_s=0.5) -> list[bytes]`**

- Output: received UDP datagram payloads (may be empty).
- Blocks up to `timeout_s` for the first datagram, then drains the rest non-blocking.

**`close() -> None`**

- Closes both sockets. Idempotent.

## Behavior

1. `connect` binds the UDP socket with `SO_REUSEADDR`, then connects TCP to the flight TC
   server.
2. `send_command` frames and signs the telecommand identically to flight ingress auth.
3. `poll_downlink` sets a receive timeout, reads one datagram, then switches to non-blocking
   drain.
4. `close` shuts down TCP and UDP regardless of prior connect state.

## Errors and faults

Raises on socket errors and misuse (send or poll before connect). Does not return `Result`
(GSE test tooling).

## Messages

None on the bus. Operates at the CCSDS byte transport layer.

## Configuration

Callers pass host, port, key, and telecommand APID explicitly. GSE harness sets these from
patched `LinkConfig` on real-link builds.

## Constraints

- RealStationLink is the TCP server. The emulator is the TCP client.
- Canonical packet builder import is `flight.libs.commands.build_tc_packet`.
- This emulator stands in for the ground segment. It is not the operational ground system.

## Related documents

- [`gse`](gse.md)
- [`gse.harness`](harness.md)
