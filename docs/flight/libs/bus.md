# flight.libs.bus

**Source:** `packages/flight/src/flight/libs/bus/`
**Kind:** package

## Purpose

The bus package provides a typed in-process pub/sub message bus. The composition root owns the
bus and injects subscriptions into apps.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`bus`](bus/bus.md) | module | `MessageBus`, `Subscription`, and queue overflow policy |

## Package interface

`flight.libs.bus` re-exports:

| Name | Kind |
| --- | --- |
| `MessageBus` | class |
| `OverflowPolicy` | enum |
| `QueuePolicy` | class |
| `Subscription` | class |

## Interactions

The composition root constructs one `MessageBus` and passes `Subscription` handles into apps.
Apps call `publish(msg)` to emit and `Subscription.get()` to receive. Routing matches the
exact message class. Subclass dispatch does not occur.

## Constraints

- Apps never construct queues themselves.
- Transport is in-process `queue.Queue`.
- `publish` puts the same object reference into every subscriber queue.
- Per-message-type overflow policy is configured by the composition root.
- Default policy is unbounded `DROP_OLDEST` when a type has no explicit entry.

## Related documents

- [`flight.libs`](../libs.md)
- [`flight.libs.bus.bus`](bus/bus.md)
- [`flight.libs.messages`](../messages.md)
