# flight.hal.drivers_real.station

**Source:** `packages/flight/src/flight/hal/drivers_real/station.py`
**Kind:** driver

## Purpose

Implements the real CCSDS station link. Inbound telecommands arrive over TCP. Outbound
telemetry and products leave over UDP. The driver satisfies `StationLink` structurally.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RealStationLink` | class | TCP-in / UDP-out CCSDS link driver |

## Inputs and outputs

Constructor:

- `cfg` (`LinkConfig`): hosts, ports, and socket timeout
- `clock` (`Clock`): injected clock, reserved for future use

Raises `ValueError` when a host is empty or a port is out of range.

Protocol methods match `StationLink`. See
[`flight.hal.interfaces.station`](../interfaces/station.md).

## Behavior

1. The constructor binds a TCP server socket on `command_tcp_host:command_tcp_port` and
   opens a UDP socket for outbound sends.
2. A daemon thread accepts one TCP client at a time and recv-loops on the stream.
3. The recv loop deframes the byte stream into complete CCSDS packets using
   `packet_length` on the 6-byte primary header.
4. Deframed packets enqueue on an inbound deque. `receive_packet` pops one packet
   non-blocking, or returns `Ok(None)` when the deque is empty.
5. `send_packet` sends one UDP datagram to
   `telemetry_udp_host:telemetry_udp_port`.
6. `link_state` returns `AOS` while a TCP client is connected and `LOS` otherwise.
7. `close` signals the thread to stop, closes both sockets, and joins the thread.
8. Invalid primary headers clear the reassembly buffer.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `FaultCode.COMM_TIMEOUT` | UDP `sendto` raises `OSError` |

`receive_packet` returns `Ok(None)` for an empty inbound queue. Library methods do not
raise at runtime.

## Messages

None.

## Configuration

Reads `LinkConfig` fields:

- `command_tcp_host`, `command_tcp_port`
- `telemetry_udp_host`, `telemetry_udp_port`
- `socket_timeout_s`

## Constraints

- One TCP client is accepted at a time.
- Sockets open at construction, not at import time.
- The driver imports `flight.libs.ccsds` for packet deframing.

## Related documents

- [`flight.hal.drivers_real`](../drivers_real.md)
- [`flight.hal.interfaces.station`](../interfaces/station.md)
