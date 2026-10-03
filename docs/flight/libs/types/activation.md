# flight.libs.types.activation

**Source:** `packages/flight/src/flight/libs/types/activation.py`
**Kind:** pure module

## Purpose

The module defines the authority-scoped activation identity shared by payload
graphs and system-mode messages. It is a pure value record: the composition
root supplies the epoch and the external mode authority owns the sequence.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ActivationKey` | dataclass | Session epoch plus activation sequence |

## Inputs and outputs

`ActivationKey` is a frozen slots dataclass with two fields: `epoch` (str),
the composition-root-provided session epoch, and `sequence` (int), the
authority-owned activation sequence within the epoch.

## Behavior

Records carry identity only. Payload state, capture contexts, command-router
epoch checks, and the system-mode message family all compare the same key
type so a stale or foreign activation can be rejected by field equality.

## Errors and faults

None.

## Messages

None. `ActivationKey` is a field type used inside bus messages and state
records, not a bus message itself.

## Configuration

None. The epoch value arrives at the composition root from configuration or
the authority handshake and is injected into each consumer.

## Constraints

The record is pure data. It never performs I/O, reads clocks, or touches the
bus.

## Related documents

- [`flight.libs.types`](../types.md)
- [`flight.libs.messages`](../messages.md)
- [`flight.payload.state`](../../payload/state.md)
