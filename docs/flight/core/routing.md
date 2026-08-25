# flight.core.routing

**Source:** `packages/flight/src/flight/core/routing.py`
**Kind:** pure module

## Purpose

The pure routing core maps one ingress-validated `CommandMsg` to a `RouteResult`. The
command router shell owns the bus, clock, and armed state.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RouteResult` | class | Per-command publish plan and updated armed map |
| `route_command` | function | Run the routing decision for one `CommandMsg` |

## Inputs and outputs

**`RouteResult` fields:**

- `routed_command`: `RoutedCommandMsg` or `None`
- `ack`: `CommandAckMsg` or `None`
- `unroutable_detail`: fault detail string or `None`
- `new_armed`: updated `(source, command_id) -> arm_time_s` map

**`route_command(command, routable_targets, hazardous_ids, safe_latched, armed, now, arm_window_s) -> RouteResult`**

- Inputs: validated command, routable target set, hazardous ID set, SAFE-latch flag, armed
  map, monotonic seconds, ARM window seconds.
- Output: `RouteResult` for the shell to publish.

## Behavior

1. When `command.target` is not in `routable_targets`, return a rejected ack and
   `unroutable_detail`. Do not dispatch.
2. When `command.target` is `"core"`, return an accepted ack with detail
   `"executed by core"`. Do not dispatch.
3. When `command.command_id` is not hazardous, return a `RoutedCommandMsg`. The target app
   emits the execution ack.
4. When hazardous and `params.phase` is `"ARM"`, record arm time in `new_armed` and return
   an accepted ack with detail `"armed"`. Do not dispatch.
5. When hazardous and `params.phase` is `"EXECUTE"`, require a prior ARM within
   `arm_window_s`. Reject when ARM is missing or expired.
6. When hazardous EXECUTE and `safe_latched` is true, reject all commands except
   `EXIT_SAFE`.
7. On valid hazardous EXECUTE, dispatch `RoutedCommandMsg` and remove the arm entry.
8. When hazardous with any other phase, reject with detail
   `"hazardous phase must be ARM/EXECUTE"`.

Returned messages carry `timestamp_utc=""`. The shell stamps timestamps before publish.

## Errors and faults

The pure core does not publish faults. It returns acks with `FaultCode.COMMAND_UNROUTABLE`
or `FaultCode.COMMAND_INVALID` in the `RouteResult`.

## Messages

None. The shell publishes messages described in the `RouteResult`.

## Configuration

The caller passes `arm_window_s` from `command_router.arm_window_s`.

## Constraints

- No bus access, no clock reads, no I/O, no logging.
- Hazardous commands use a two-step ARM then EXECUTE sequence.
- `EXIT_SAFE` is the only hazardous command allowed while SAFE-latched.

## Related documents

- [`flight.core`](../core.md)
- [`flight.core.command_router`](command_router.md)
