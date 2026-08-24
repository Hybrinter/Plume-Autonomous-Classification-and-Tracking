# flight.libs.bus

**Source:** `packages/flight/src/flight/libs/bus`
**Kind:** package

## Purpose

This package provides the typed in-process message bus used by all subsystem apps.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`bus`](flight/libs/bus/bus.md) | module | `MessageBus`, `Subscription`, queue policies |

## Package interface

Re-exports from `flight.libs.bus.bus`:

- `MessageBus`
- `OverflowPolicy`
- `QueuePolicy`
- `Subscription`

## Interactions

The composition root constructs a `MessageBus` and injects `Subscription` handles into
apps. Apps call `publish` with message dataclasses from `flight.libs.messages`. Apps call
`Subscription.get` or `get_nowait` to receive messages.

## Constraints

- Routing matches the exact message type. Subclasses do not match a base-type subscription.
- `publish` enqueues the same object reference to every subscriber queue.
- The composition root owns bus construction. Apps do not create their own queues.

## Related documents

- [`flight.libs`](flight/libs.md)
- [`flight.libs.bus.bus`](flight/libs/bus/bus.md)
- [`flight.libs.messages`](flight/libs/messages.md)
