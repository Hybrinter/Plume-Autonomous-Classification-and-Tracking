# flight.hal.drivers_real.station

**Source:** `packages/flight/src/flight/hal/drivers_real/station.py`
**Kind:** driver

## Purpose

`RealStationLink` implements the station data link with CCSDS telecommands over TCP and
telemetry over UDP. It satisfies `StationLink` structurally.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RealStationLink` | class | TCP-in / UDP-out CCSDS link driver |

## Inputs and outputs

Construction takes `LinkConfig` and a `Clock`.

| Method | Inputs | Outputs |
| --- | --- | --- |
| `receive_packet()` | None | `Result[bytes \| None, FaultCode]` |
| `send_packet(packet)` | Complete CCSDS packet bytes | `Result[None, FaultCode]` |
| `link_state()` | None | `LinkState` |
| `close()` | None | None |

Construction raises `ValueError` on empty hosts or ports outside 1..65535.

## Behavior

1. Construction binds a TCP server on the command host and port and opens a UDP socket for
   outbound telemetry.
2. A daemon thread accepts one station client and recv-loops on the TCP stream.
3. The recv loop deframes the byte stream into complete CCSDS packets using
   `flight.libs.ccsds.packet_length` on the 6-byte primary header.
4. Deframed packets enqueue on an internal deque. `receive_packet()` pops one packet
   non-blockingly or returns `Ok(None)`.
5. `send_packet()` sends one UDP datagram to the telemetry endpoint.
6. `link_state()` returns `AOS` while a TCP client is connected and `LOS` otherwise.
7. `close()` signals the thread, closes both sockets, and joins the thread.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `COMM_TIMEOUT` | Outbound UDP `sendto` fails |

Malformed inbound headers clear the deframe buffer inside the recv loop. They do not surface
as `Err` on `receive_packet()`.

## Messages

None.

## Configuration

Reads `LinkConfig`: command TCP host and port, telemetry UDP host and port, and socket
timeout. APID fields are used above this layer.

## Constraints

- Uses stdlib sockets only. No vendor SDK.
- Sockets open at construction. The recv thread starts immediately.
- `close()` is idempotent.
- Library methods return `Result` and do not raise on runtime faults.

## Related documents

- [`flight.hal.interfaces.station`](interfaces/station.md)
- [`flight.hal.drivers_real`](drivers_real.md)
- [`flight.hal.drivers_sim.station`](drivers_sim/station.md)
- [`flight.libs.ccsds`](libs/ccsds.md)
