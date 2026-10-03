# flight.payload.graphs

**Source:** `packages/flight/src/flight/payload/graphs`
**Kind:** package

## Purpose

The graphs package holds the pure payload mode graphs: the `IDLE`, `STOW`,
`SAFE`, `INIT`, and `OPERATE` graph implementations, their shared typed
contracts in `base`, the config projection in `parameters`, and the closed
`runtime` dispatch. The package `__init__` carries no exports and no mode
mapping.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`base`](graphs/base.md) | pure module | Typed graph specs, policies, inputs, outcomes, and activation helpers |
| [`parameters`](graphs/parameters.md) | pure module | `GraphParameters` config projection and feedback freshness |
| [`idle`](graphs/idle.md) | pure module | One-node pose-hold IDLE graph |
| [`safe`](graphs/safe.md) | pure module | One-node inhibition-only SAFE graph |
| [`stow`](graphs/stow.md) | package | Bounded move plus verified hold |
| [`init`](graphs/init.md) | package | Effect chain plus verification-wait readiness |
| [`operate`](graphs/operate.md) | package | Tracking, hunts, and hold with command edges |
| [`runtime`](graphs/runtime.md) | pure module | Closed-union dispatch over the five graphs |

## Package interface

Each graph module or package exposes `spec(params)`, `initial_state(inputs,
params)`, and `step(state, inputs, params)`. `runtime` dispatches over the
closed state union and applies commands only on OPERATE.

## Interactions

`PayloadApp` is the live shell: each accepted `SystemModeActivatedMsg`
reenters the graph selected by the external authority, `control_tick` steps
the installed graph, and graph outcomes commit through the shell's HAL and
bus surfaces. The graphs themselves stay pure - no I/O, bus, or clock - and
never import `SystemMode`; the mode-to-graph mapping lives only in the app
shell. Shared gimbal and tracking primitives never import this package.

## Constraints

All modules are pure: no I/O, no bus access, no clock reads, no `SystemMode`
imports, no callable dispatch tables, and no general statechart machinery.
Guard exclusivity lives in the concrete graphs; the base layer validates only
topology and directed command edges.

## Related documents

- [`flight.payload`](../payload.md)
- [`flight.payload.records`](records.md)
- [`flight.payload.gimbal.request`](gimbal/request.md)
