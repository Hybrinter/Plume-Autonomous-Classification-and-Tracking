# flight.libs.bus.bus

**Source:** `packages/flight/src/flight/libs/bus/bus.py`
**Kind:** module

## Purpose

The module implements a typed in-process pub/sub message bus with per-message-type queue bounds
and overflow policy.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `OverflowPolicy` | enum | `DROP_OLDEST` or `NEVER_DROP` |
| `QueuePolicy` | class | Per-type `maxsize` and `overflow` settings |
| `Subscription` | class | Typed receive handle for one message type |
| `MessageBus` | class | Pub/sub bus routed by exact message type |

### Subscription methods

| Name | Kind | Description |
| --- | --- | --- |
| `get(timeout)` | method | Block for the next message |
| `get_nowait()` | method | Return the next message or raise `queue.Empty` |
| `empty()` | method | Return True when no message is queued |

### MessageBus methods

| Name | Kind | Description |
| --- | --- | --- |
| `subscribe(message_type)` | method | Register interest and return a `Subscription` |
| `publish(message)` | method | Deliver to all subscribers of `type(message)` |
| `dropped_count(message_type)` | method | Count of dropped messages for `DROP_OLDEST` types |
| `overflow_count(message_type)` | method | Soft-bound exceedance count for `NEVER_DROP` types |
| `total_dropped()` | method | Sum of all dropped counts |
| `total_overflow()` | method | Sum of all overflow counts |
| `queue_depth(message_type)` | method | Total queued messages across subscribers |

## Inputs and outputs

| Entry point | Inputs | Outputs |
| --- | --- | --- |
| `MessageBus(policy)` | Optional `dict[type, QueuePolicy]` | Empty bus instance |
| `subscribe(T)` | Message class `T` | `Subscription[T]` |
| `publish(msg)` | Message instance | None; delivers to matching subscribers |
| `Subscription.get(timeout)` | Optional timeout seconds | Message instance |
| `queue_depth(T)` | Message class `T` | Integer backlog count |

## Behavior

1. The composition root constructs the bus with an optional per-type policy map.
2. Each app calls `subscribe(MessageClass)` and receives a typed `Subscription`.
3. `publish(msg)` routes by exact `type(msg)`. Subclass matching does not occur.
4. For `DROP_OLDEST` with `maxsize > 0`, overflow discards the oldest queued message and
   increments the drop counter.
5. For `NEVER_DROP` with `maxsize > 0`, the queue stays unbounded. Exceeding the soft bound
   increments the overflow counter. No message is discarded.
6. When `maxsize` is 0, the queue is unbounded with no overflow accounting.
7. `queue_depth()` sums `qsize()` across subscriber queues without consuming messages.

## Errors and faults

None from the bus itself. Overflow is counted via `dropped_count` and `overflow_count`.

`Subscription.get()` and `get_nowait()` raise `queue.Empty` when no message is available.

## Messages

The bus carries all types defined in `flight.libs.messages`. It does not define message types.

## Configuration

The composition root supplies the per-type `QueuePolicy` map at construction. The bus imports
no message classes.

## Constraints

- Apps never construct queues. The composition root owns the bus.
- `publish` delivers the same object reference to every subscriber queue.
- Consumers must treat received messages as immutable.
- Default policy for unlisted types is unbounded `DROP_OLDEST` (`maxsize=0`).
- Transport is in-process `queue.Queue`.

## Related documents

- [`flight.libs.bus`](../bus.md)
- [`flight.libs.messages`](../messages.md)
