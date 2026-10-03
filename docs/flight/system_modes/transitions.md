# flight.system_modes.transitions

**Source:** `packages/flight/src/flight/system_modes/transitions.py`
**Kind:** pure module

## Purpose

The transitions module holds the system-mode transition table and the `decide` function. The
authority shell calls `decide` once for each request.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RequestKind` | enum | Request origin: `SUBSYSTEM`, `SET_MODE`, or `EXIT_SAFE` |
| `ModeRequest` | class | Frozen requested target mode plus `RequestKind` |
| `SafetyEvidence` | class | Frozen latest fault latch flag and active SAFE-triggering faults |
| `Decision` | class | Frozen decision, resulting mode, reason, and `recovery_authorized` |
| `SYSTEM_MODES` | constant | `frozenset` of the five system modes |
| `MODE_EDGES` | constant | `frozenset` of allowed `(current, kind, target)` edges |
| `decide` | function | Maps current mode, request, and evidence to a `Decision` |

## Inputs and outputs

- `decide(current, request, safety)` takes the active mode (or `None` before the first
  activation), a `ModeRequest`, and a `SafetyEvidence`. It returns a `Decision`.
- An accepted `Decision` carries the new mode. A denied `Decision` keeps `current`.

## Behavior

`decide` applies these rules in order:

1. Deny a target that is not in `SYSTEM_MODES`.
2. Deny a request for the current mode.
3. Accept `SAFE` from every other state and every request kind.
4. In `SAFE`, accept only `EXIT_SAFE` to `INIT`, and only while `active_faults` is empty. That
   decision sets `recovery_authorized`.
5. Deny `EXIT_SAFE` outside `SAFE`.
6. Deny every other transition while `safe_latched` is true.
7. Accept an edge in `MODE_EDGES`. Deny all other requests with a reason.

`MODE_EDGES`:

| Current | Kind | Target |
| --- | --- | --- |
| `INIT` | `SUBSYSTEM` | `IDLE` |
| `IDLE` | `SET_MODE` | `INIT`, `OPERATE`, `STOW` |
| `OPERATE` | `SET_MODE` | `IDLE`, `STOW` |
| `STOW` | `SET_MODE` | `IDLE` |

With no active mode, only `SAFE` is accepted, so the system boots into `SAFE`. `INIT` is entered
only by a ground command: `EXIT_SAFE` from `SAFE` or `SET_MODE` from `IDLE`. `INIT` hands off to
`IDLE` on a subsystem request once the subsystems report ready. The readiness signal is not
defined yet.

## Errors and faults

None. Every input yields a `Decision`. Denials carry a reason string.

## Messages

None. The module does not build or publish messages.

## Configuration

None.

## Constraints

- Pure module with no I/O, bus access, clock reads, or ID generation.
- The legacy `SystemMode` members `ACTIVE`, `SCAN`, `MODEL_UPLINK`, and `DATA_DOWNLINK` are
  never activated.

## Related documents

- [`flight.system_modes`](../system_modes.md)
- [`flight.system_modes.app`](app.md)
