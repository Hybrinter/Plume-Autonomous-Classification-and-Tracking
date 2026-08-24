# flight.libs.bus.bus

**Source:** `packages/flight/src/flight/libs/bus/bus.py`
**Kind:** module

## Purpose

This module implements a typed pub/sub bus with per-message-type queue bounds and overflow
policies. Transport uses in-process `queue.Queue`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `OverflowPolicy` | enum | `DROP_OLDEST` or `NEVER_DROP` |
| `QueuePolicy` | dataclass | `maxsize` and `overflow` for one message type |
| `Subscription` | class | Typed receive handle for one message type |
| `MessageBus` | class | Pub/sub bus with policy map and counters |

### `Subscription`

| Name | Kind | Description |
| --- | --- | --- |
| `get` | method | Block for the next message; optional timeout |
| `get_nowait` | method | Return the next message or raise `queue.Empty` |
| `empty` | method | Return whether the queue has no messages |

### `MessageBus`

| Name | Kind | Description |
| --- | --- | --- |
| `subscribe` | method | Register a message type; return a `Subscription` |
| `publish` | method | Deliver a message to all subscribers of its type |
| `dropped_count` | method | DROP_OLDEST discard count for one type |
| `overflow_count` | method | NEVER_DROP soft-bound exceedance count for one type |
| `total_dropped` | method | Sum of dropped counts across types |
| `total_overflow` | method | Sum of overflow counts across types |
| `queue_depth` | method | Sum of queued messages across subscribers of one type |

## Inputs and outputs

- `MessageBus(policy=None)` accepts an optional `dict[type, QueuePolicy]`. It returns a bus
  instance.
- `subscribe(message_type)` returns a `Subscription` for that exact type.
- `publish(message)` delivers `message` to every queue registered for `type(message)`.
- Counter and depth methods take a message type and return an integer.

## Behavior

1. A missing policy entry uses an unbounded queue with `DROP_OLDEST` semantics and
   `maxsize=0`.
2. On `subscribe`, a `DROP_OLDEST` policy creates a bounded queue with hard `maxsize`. A
   `NEVER_DROP` policy creates an unbounded queue.
3. On `publish`, the bus looks up subscribers for the message's exact runtime type.
4. For unbounded policy (`maxsize <= 0`), the bus always enqueues the message.
5. For `DROP_OLDEST` with a positive bound, a full queue drops the oldest message,
   increments the per-type drop counter, then enqueues the new message.
6. For `NEVER_DROP` with a positive soft bound, the bus never discards. When
   `qsize >= maxsize`, it increments the per-type overflow counter, then enqueues.
7. `queue_depth` sums `qsize()` across subscriber queues without consuming messages.

## Errors and faults

None. Methods do not return `Result`. `Subscription.get` and `get_nowait` raise
`queue.Empty` when no message is available.

## Messages

The bus carries any published object. Flight code publishes dataclasses from
`flight.libs.messages`. The bus module imports no message classes.

## Configuration

The composition root passes an optional `dict[type, QueuePolicy]` at construction. The bus
does not read TOML or config dataclasses directly.

## Constraints

- Delivery matches exact types only.
- `publish` shares one object reference across subscriber queues.
- Policy keys are message types. The bus does not import message modules.

## Related documents

- [`flight.libs.bus`](flight/libs/bus.md)
- [`flight.libs.messages`](flight/libs/messages.md)
