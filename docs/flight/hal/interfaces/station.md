# flight.hal.interfaces.station

**Source:** `packages/flight/src/flight/hal/interfaces/station.py`
**Kind:** module

## Purpose

Defines the station byte-link Protocol. Inbound telecommands arrive as raw CCSDS packet
bytes. Outbound telemetry and products leave as raw CCSDS packet bytes. Framing, CRC,
authentication, and validation live in `iss_iface`, not in the link driver.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `StationLink` | class | Runtime-checkable Protocol for the ISS/station transport |

## Inputs and outputs

| Method | Inputs | Output |
| --- | --- | --- |
| `receive_packet` | none | `Result[bytes \| None, FaultCode]` |
| `send_packet` | `packet` (bytes) | `Result[None, FaultCode]` |
| `link_state` | none | `LinkState` |
| `close` | none | none |

## Behavior

1. `receive_packet` pops the next complete inbound CCSDS packet, or returns `Ok(None)`
   when none is pending.
2. `send_packet` transmits one complete outbound CCSDS packet.
3. `link_state` returns the current acquisition state (`AOS` or `LOS`).
4. `close` releases sockets and threads. The call is safe to repeat.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `FaultCode.COMM_TIMEOUT` | Outbound send fails (real driver only) |

`receive_packet` returns `Ok(None)` when the inbound queue is empty. It does not signal
a fault for an empty queue.

## Messages

None.

## Configuration

None at the Protocol level. The real driver reads `LinkConfig` at construction.

## Constraints

- The link is byte-level transport only.
- `link_state` does not return a `Result`. It returns a `LinkState` enum member
  directly.

## Related documents

- [`flight.hal.interfaces`](../interfaces.md)
- [`flight.hal.drivers_real.station`](../drivers_real/station.md)
- [`flight.hal.drivers_sim.station`](../drivers_sim/station.md)
