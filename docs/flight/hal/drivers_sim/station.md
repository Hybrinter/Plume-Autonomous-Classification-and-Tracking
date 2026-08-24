# flight.hal.drivers_sim.station

**Source:** `packages/flight/src/flight/hal/drivers_sim/station.py`
**Kind:** driver

## Purpose

`SimStationLink` replays scripted inbound CCSDS packets and records outbound packets. It
satisfies `StationLink` structurally for SIL and tests.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SimStationLink` | class | Scripted CCSDS packet replay and record driver |
| `SimStationLink.sent` | property | Outbound packets in send order (test hook) |
| `SimStationLink.enqueue(packet)` | method | Append a later inbound packet |
| `SimStationLink.set_link_state(state)` | method | Set reported AOS/LOS state |

## Inputs and outputs

Construction takes an optional inbound `list[bytes]` and a `LinkState` (default `AOS`).

| Method | Inputs | Outputs |
| --- | --- | --- |
| `receive_packet()` | None | `Result[bytes \| None, FaultCode]` |
| `send_packet(packet)` | Complete CCSDS packet | `Ok(None)` |
| `link_state()` | None | `LinkState` |
| `close()` | None | None (no-op) |

## Behavior

1. Each `receive_packet()` call returns the next scripted inbound packet.
2. After the inbound list is exhausted, `receive_packet()` returns `Ok(None)`.
3. `send_packet()` appends the packet to an internal sent list and succeeds.
4. `link_state()` returns the configured state until `set_link_state()` changes it.
5. `enqueue()` appends packets for later receive calls during a test or SIL run.
6. `close()` is a no-op.

## Errors and faults

None under normal operation.

## Messages

None.

## Configuration

None. Inbound packets and initial link state are supplied at construction or via
`enqueue()` and `set_link_state()`.

## Constraints

- End-of-script behavior returns an empty inbound queue (`Ok(None)`), not a fault.
- The `sent` property is a test and SIL inspection hook. No real driver exposes it.
- No sockets or background threads are used.

## Related documents

- [`flight.hal.interfaces.station`](interfaces/station.md)
- [`flight.hal.drivers_sim`](drivers_sim.md)
- [`flight.hal.drivers_real.station`](drivers_real/station.md)
