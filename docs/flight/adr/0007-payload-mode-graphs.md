# ADR-FLIGHT-0007: Payload mode graphs own payload behavior

**Status:** Accepted
**Date:** 2026-10-02
**Topic:** restructure
**Supersedes:** ADR-FLIGHT-0005 (GimbalState arbiter model), ADR-FLIGHT-0002 (SAFE stow), ADR-REPO-0008 (SAFE stow)
**Superseded-by:** none
**Related:** ADR-FLIGHT-0004, ADR-FLIGHT-0005

## Context

The payload controller mixes mode dispatch, hunting policy, residual replay, and
scene selection across `ControlState`, `GimbalArbiter`, `outer_rate`, and
`select_scene`. Shared gimbal helpers dispatch on the payload-local
`GimbalState` enum (`outer_rate(mode, ...)`, `select_scene(mode, ...)`), and the
arbiter owns both limb-wait timing and mode identity. Adding commands, hold
semantics, and per-mode imaging policy on that structure keeps coupling payload
behavior to mode-bearing dispatch inside shared primitives.

System activation is owned by a teammate's mode authority. Payload behavior must
be selectable by an authoritative activation without payload code branching on
`SystemMode` or importing the authority.

## Decision

- Define five payload graphs named after the system modes (`IDLE`, `STOW`,
  `SAFE`, `INIT`, `OPERATE`). The system authority chooses a mode; the payload
  shell maps it to the name-mirrored graph at one boundary. Graph and node
  functions never branch on or import `SystemMode`.
- Graphs are pure functions over explicit inputs: typed specs, immutable
  directed edges, per-graph node enums, a closed `GraphState` union, and typed
  `NodeOutcome`/`GraphOutcome` results. No clocks, logging, hardware, bus, or
  threads live inside a graph. Use file-per-node organization; no generic
  hierarchical statechart machinery.
- Shared gimbal and tracking helpers become mode-free primitives. References
  form a closed union (`RateReference`, `PoseReference`, `StowReference`,
  `InhibitReference`) carrying explicit travel envelopes, not mode names.
- Graphs own imaging and inference policy: `IDLE`/`STOW`/`SAFE` disable
  acquisition; `INIT` captures only for explicitly requested verification work;
  `OPERATE` keeps current defaults with typed overrides.
- Commands commit only across declared directed edges. `OPERATE` distinguishes
  automatic limb-wait `HOLD` (vision may reacquire) from manual `HOLD`
  (explicit `GIMBAL_RESUME` required). `SAFE` emits only inhibit references;
  containment stays a separate authority from graph selection.
- Runtime implementation is pending: this record approves the architecture, not
  an as-built change. Teammate agreement on the shared interface is a pending
  external dependency.

## Consequences

- Mode-bearing helper signatures are removed in PR 2; `gimbal/arbiter.py` and
  `GimbalState` are removed at the PR 5 cutover; node enums move into their
  graphs.
- One activation-driven runtime replaces the current mode-branching controller;
  no legacy aliases or competing live runtime remain.
- `INIT` requests `IDLE` only after verified stability; the stability criterion
  stays deferred with a pending production default.

## Alternatives considered

- Keep the arbiter and add modes -- preserves the mode/policy coupling that this
  refactor removes.
- A generic FSM/statechart library -- rejected as unneeded machinery; typed
  closed unions and explicit `match` cover five graphs.
