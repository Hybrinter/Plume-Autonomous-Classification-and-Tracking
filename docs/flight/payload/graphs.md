# flight.payload.graphs

**Source:** `packages/flight/src/flight/payload/graphs`
**Kind:** package

## Purpose

The graphs package holds the typed pure contracts for payload mode graphs:
specs, directed edges, imaging and inference policies, tick inputs, outcomes,
effect identities, and activation-key bookkeeping. The package `__init__`
carries no exports and no mode mapping.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`base`](graphs/base.md) | pure module | Typed graph specs, policies, inputs, outcomes, and activation helpers |

## Package interface

`flight.payload.graphs.base` defines `GraphId`, `Edge`, `GraphSpec`,
`EdgeTrigger`, policy records and limits, `resolve_policy`, `validate_policy`,
`validate_spec`, `command_target`, effect and outcome records, `TickInputs`,
and `accept_activation`/`ActivationState`/`ActivationSnapshot`.

## Interactions

These contracts are inert foundations: no module here executes a graph step or
drives hardware, and no shell wiring to them exists yet. Activation mapping at
the shell boundary arrives with the runtime cutover. Shared gimbal and
tracking primitives never import this package.

## Constraints

All modules are pure: no I/O, no bus access, no clock reads, no `SystemMode`
imports, no callable dispatch tables, and no general statechart machinery.

## Related documents

- [`flight.payload`](../payload.md)
- [`flight.payload.records`](records.md)
- [`flight.payload.gimbal.request`](gimbal/request.md)
