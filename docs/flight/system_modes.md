# flight.system_modes

**Source:** `packages/flight/src/flight/system_modes`
**Kind:** subsystem app

## Purpose

The system_modes package is the system-mode authority. It owns the active system mode. It decides
each mode request and each system-mode ground command against one explicit transition table. It
publishes a transition record for every decision and an activation for every accepted transition.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`app`](system_modes/app.md) | module | Authority app shell: bus I/O, activation keys, ACKs, sync replies |
| [`transitions`](system_modes/transitions.md) | pure module | System modes, transition table, and `decide` |

## Package interface

`flight.system_modes.__init__` is empty. Import `SystemModesApp` from `flight.system_modes.app`
and the transition types from `flight.system_modes.transitions`.

## Interactions

The authority subscribes to `SystemModeRequestMsg`, `RoutedCommandMsg`, `SafetyStateMsg`, and
`SystemModeSyncRequestMsg`. It publishes `SystemModeTransitionMsg`, `SystemModeActivatedMsg`,
`CommandAckMsg`, and `HeartbeatMsg`. The command router routes `SET_MODE` and `EXIT_SAFE` to the
`system_modes` target. The fault app requests SAFE and releases its SAFE latch only on a
recovery-authorized activation. The startup health gate requests INIT or SAFE. The composition
root supplies the session epoch. The authority does not use HAL drivers.

## Constraints

- The system modes are `IDLE`, `STOW`, `SAFE`, `INIT`, and `OPERATE`.
- `transitions` is pure: no bus, clock, ID generation, or I/O.
- Requests and transition records never select behavior. Only `SystemModeActivatedMsg` does.
- Fault containment does not wait for the authority. The fault app latches SAFE first.
- The payload does not consume activations yet. The payload keeps its `ModeChangeMsg` input.

## Related documents

- [`flight.fault`](fault.md)
- [`flight.core.composition`](core/composition.md)
- [`flight.core.routing`](core/routing.md)
- [`flight.libs.messages.messages`](libs/messages/messages.md)
