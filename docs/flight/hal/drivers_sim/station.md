# flight.hal.drivers_sim.station

**Source:** `packages/flight/src/flight/hal/drivers_sim/station.py`
**Kind:** driver

## Purpose

Replays scripted inbound CCSDS packets and records outbound packets for SIL and tests.
The driver satisfies `StationLink` structurally.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SimStationLink` | class | Scripted packet replay and recording link |
| `SimStationLink.enqueue` | function | Appends an inbound packet to the replay queue |
| `SimStationLink.set_link_state` | function | Sets the reported `LinkState` |
| `SimStationLink.sent` | property | Tuple of all outbound packets, in order |

## Inputs and outputs

Constructor:

- `inbound` (`list[bytes] | None`): CCSDS packets to replay; default empty list
- `link_state` (`LinkState`): fixed acquisition state; default `AOS`

| Method | Inputs | Output |
| --- | --- | --- |
| `receive_packet` | none | `Result[bytes \| None, FaultCode]` |
| `send_packet` | `packet` (bytes) | `Result[None, FaultCode]` |
| `link_state` | none | `LinkState` |
| `close` | none | none |

## Behavior

1. `receive_packet` returns the next inbound packet and advances an internal index.
2. When the inbound list is exhausted, `receive_packet` returns `Ok(None)`.
3. `send_packet` appends the packet to an internal sent list and returns `Ok(None)`.
4. `link_state` returns the scripted state set at construction or by
   `set_link_state`.
5. `enqueue` appends a packet for a later `receive_packet` call.
6. `close` is a no-op.
7. The `sent` property exposes all outbound packets for test inspection.

## Errors and faults

None. All methods return `Ok`.

## Messages

None.

## Configuration

None. Inbound packets and link state are supplied at construction or via test hooks.

## Constraints

- End-of-script inbound behavior returns `Ok(None)`, matching an empty inbound queue.
- `sent` has no counterpart on the real driver.
- `enqueue` and `set_link_state` are test and SIL hooks.

## Related documents

- [`flight.hal.drivers_sim`](../drivers_sim.md)
- [`flight.hal.interfaces.station`](../interfaces/station.md)
