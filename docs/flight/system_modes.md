# flight.system_modes

**Source:** `packages/flight/src/flight/system_modes`
**Kind:** subsystem app

## Purpose

The system_modes package is the system-mode authority. It owns the active system mode. It
decides each mode request and each system-mode ground command against one explicit transition
table. It publishes a transition record for every decision and an activation for every
accepted transition.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`app`](system_modes/app.md) | app shell | Authority app: bus I/O, activation keys, ACKs, sync replies |
| [`transitions`](system_modes/transitions.md) | pure module | System modes, transition table, and `decide` |

## Package interface

`flight.system_modes.__init__` contains only the package docstring; it has no
public reexports. Import `SystemModesApp` from `flight.system_modes.app`
and the transition types from `flight.system_modes.transitions`.

## Interactions

The authority subscribes to `SystemModeRequestMsg`, `RoutedCommandMsg`, `SafetyStateMsg`, and
`SystemModeSyncRequestMsg`. It publishes `SystemModeTransitionMsg`, `SystemModeActivatedMsg`,
`CommandAckMsg`, and `HeartbeatMsg`. The command router routes `SET_MODE`, `EXIT_SAFE`, and
`GIMBAL_STOW` to the `system_modes` target. The fault app requests SAFE and releases its SAFE
latch only on a recovery-authorized `SAFE -> INIT` activation. The payload consumes
activations and releases its own containment latch under the same authorized-recovery
contract; it requests `INIT -> IDLE` only from its verified INIT lifecycle completion, keyed
to the current activation. The authority boots into SAFE. The startup health gate requests
SAFE when it fails. The composition root supplies the session epoch. The authority does not
use HAL drivers.

## Constraints

- The system modes are `IDLE`, `STOW`, `SAFE`, `INIT`, and `OPERATE`.
- `transitions` is pure: no bus, clock, ID generation, or I/O.
- Requests and transition records never select behavior. Only `SystemModeActivatedMsg` does.
- Fault containment does not wait for the authority. The fault app latches SAFE first.
- `system_modes` is a peer app: it never imports another app, and no app imports it.

## Related documents

- [`flight.fault`](fault.md)
- [`flight.payload`](payload.md)
- [`flight.core.composition`](core/composition.md)
- [`flight.core.routing`](core/routing.md)
- [`flight.libs.messages.messages`](libs/messages/messages.md)
