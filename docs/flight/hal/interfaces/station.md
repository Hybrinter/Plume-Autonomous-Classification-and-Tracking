# flight.hal.interfaces.station

**Source:** `packages/flight/src/flight/hal/interfaces/station.py`
**Kind:** module

## Purpose

This module defines the `StationLink` Protocol for byte-level CCSDS Space Packet transport
between the payload and the station. Inbound telecommands arrive as raw bytes. Outbound
telemetry and products leave as raw bytes.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `StationLink` | Protocol | Byte-level station data link |

## Inputs and outputs

| Method | Inputs | Outputs |
| --- | --- | --- |
| `receive_packet()` | None | `Result[bytes \| None, FaultCode]` |
| `send_packet(packet)` | One complete CCSDS packet | `Result[None, FaultCode]` |
| `link_state()` | None | `LinkState` (`AOS` or `LOS`) |
| `close()` | None | None |

`Ok(None)` from `receive_packet()` means no inbound packet is pending.

## Behavior

1. The iss_iface app polls `receive_packet()` for inbound telecommand bytes.
2. The app passes complete outbound packets to `send_packet()`.
3. `link_state()` reports acquisition state. The app gates downlink draining on `AOS`.
4. `close()` releases sockets and background threads. It is safe to call more than once.

Framing, CRC, authentication, and validation live in iss_iface, not in the link driver.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `COMM_TIMEOUT` | Outbound UDP send fails (real driver) |

Inbound deframing errors are handled inside concrete drivers. The Protocol does not fix
every fault code.

## Messages

None.

## Configuration

None at the Protocol level. The real driver reads hosts, ports, and timeouts from
`LinkConfig`.

## Constraints

- The link is pure byte transport. It does not parse command opcodes or telemetry schemas.
- CCSDS packet assembly and disassembly happen above this layer in iss_iface and
  `flight.libs.ccsds`.

## Related documents

- [`flight.hal.interfaces`](interfaces.md)
- [`flight.hal.drivers_real.station`](drivers_real/station.md)
- [`flight.hal.drivers_sim.station`](drivers_sim/station.md)
- [`flight.iss_iface`](iss_iface.md)
