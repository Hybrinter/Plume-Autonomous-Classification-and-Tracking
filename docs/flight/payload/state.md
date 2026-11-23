# flight.payload.state

**Source:** `packages/flight/src/flight/payload/state.py`
**Kind:** pure module

## Purpose

The module defines `PayloadState`, the threaded runtime record owned by the
payload shell's control path. It bundles the mode-free servo memory, the
activation-acceptance state, the live graph state, the last committed control
reference and effective policy, and the shell revision tokens that invalidate
queued capture work. The INIT effect executor is not implemented in this cutover.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `PayloadState` | dataclass | Servo, activation, graph, reference, policy, and revisions |
| `graph_id_of` | function | Accepted `GraphId` or `None` without an activation |
| `node_of` | function | Live graph's node enum or `None` when no graph is installed |
| `graph_name_of` | function | Accepted graph id value, `""` when unavailable |
| `node_name_of` | function | Live node value, `""` when no graph is installed |
| `unactivated_reference` | function | Boot `InhibitReference` before any activation |

## Inputs and outputs

`PayloadState` is a frozen slots dataclass. Fields: `servo` (`ServoState`),
`activation` (`ActivationState`), `graph` (`GraphState | None`),
`reference` (`ControlReference`), `policy` (`EffectivePolicy`),
`last_outer_s` (`float | None`), `policy_revision` (int, bumped on every
accepted activation or policy change), and `control_revision` (int, bumped on
every accepted activation).

The graph helpers gate on the activation snapshot: `graph_id_of` reports
`None` when no activation is accepted, even if a graph record is installed.
The node helpers gate only on `graph`: `node_of` and `node_name_of` read the
live node's enum directly. `unactivated_reference()` returns
`InhibitReference("unactivated")`.

## Behavior

The shell threads one `PayloadState` through each control tick and swaps it
under the state lock when a tick commits. Queued capture and detection work
carries a `CaptureContext`; a bumped `policy_revision` or containment
generation marks that context stale so superseded work is dropped. Before
the first accepted activation, `graph` is `None` and `reference` is the boot
inhibit.

## Errors and faults

None. The helpers return `None`/`""` and never raise.

## Messages

None.

## Configuration

None.

## Constraints

The records are pure data. They never perform I/O, read clocks, or touch the
bus. Pure graphs stay mode-free: no `SystemMode` import lives here; the shell
maps activations onto graphs.

## Related documents

- [`flight.payload`](../payload.md)
- [`flight.payload.app`](app.md)
- [`flight.payload.graphs.runtime`](graphs/runtime.md)
- [`flight.libs.types.activation`](../libs/types/activation.md)
