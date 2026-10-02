# flight.payload.graphs.base

**Source:** `packages/flight/src/flight/payload/graphs/base.py`
**Kind:** pure module

## Purpose

The module declares the typed contracts every payload graph shares: node/edge
topology, imaging and inference policy vocabulary, tick inputs, outcomes,
effect identities, and activation-key acceptance.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `GraphId` | enum | `IDLE`/`STOW`/`SAFE`/`INIT`/`OPERATE` with lowercase values |
| `EdgeTrigger` | enum | Explicit edge triggers including `COMMAND`, `EFFECT_COMPLETED`, `VERIFIED_STABLE` |
| `ModelSelection` | enum | Typed model selection; `CONFIGURED` only |
| `EffectKind`, `EffectStatus`, `InitVerificationStatus`, `SystemRequestIntent`, `ActivationDisposition` | enums | Effect, verification, request-intent, and disposition vocabulary |
| `Edge` | dataclass | Directed edge: source, target, trigger, optional `CommandId` |
| `ImagingPolicy`, `InferencePolicy` | dataclasses | Complete acquisition/inference policies |
| `PolicyLimits`, `EffectivePolicy` | dataclasses | Sensor bounds and resolved policy pair |
| `ImagingOverride`, `InferenceOverride` | dataclasses | Optional per-node field overrides |
| `GraphSpec` | dataclass | Graph identity, nodes, initial node, edges, default policies |
| `EffectIntent`, `EffectResult`, `InitVerificationResult` | dataclasses | Activation-scoped effect and verification records |
| `TickInputs` | dataclass | Explicit observations for one graph tick |
| `NodeOutcome`, `GraphOutcome`, `CommandOutcome` | dataclasses | Typed step and command results |
| `ActivationSnapshot`, `ActivationState`, `ActivationDecision` | dataclasses | Activation contents, consumer state, and classification |
| `validate_policy` | function | `Result[None, FaultCode]` policy validation against limits |
| `resolve_policy` | function | `Result[EffectivePolicy, FaultCode]` from defaults plus overrides |
| `validate_spec` | function | `Result[None, FaultCode]` topology validation |
| `command_target` | function | `Result[NodeT, FaultCode]` directed command-edge resolution |
| `accept_activation` | function | `Result[ActivationDecision, FaultCode]` activation classification |

## Inputs and outputs

Validators and resolvers take plain values and return `Result`. `TickInputs`
carries monotonic time, an ISO timestamp, the activation key, encoder,
navigation, scoped vision, health, an optional routed command candidate,
effect results, an optional verification result, and stow-completion evidence.

## Behavior

1. `validate_spec` rejects empty or duplicated nodes, an undeclared initial
   node, undeclared edge endpoints, exact duplicate edges, command edges
   lacking a `command_id`, non-command edges carrying one, and multiple
   command destinations sharing (source, command_id). Distinct-target
   automatic edges sharing (source, trigger) are legal; the concrete graph
   owns guard exclusivity and selects its directed target edge itself.
2. `command_target` validates the spec, requires a declared source node, and
   returns the single matching directed command edge target.
3. `validate_policy` enforces finite values, ordered positive limits, exact
   positive-int decimation (bool rejected), duty in [0, 1], exposure and gain
   bounds, the camera frame-rate period, and exposure fitting the interval.
   Enabled inference requires enabled acquisition and nonzero duty.
4. `resolve_policy` applies override fields over graph defaults with
   `dataclasses.replace` and validates the complete result.
5. `accept_activation` rejects empty or unexpected epochs, negative sequences,
   and conflicting contents under an existing key. The first snapshot accepts;
   identical contents under the same key are duplicates; a lower sequence is
   stale; a newer sequence is accepted and reports a gap when it skips.

## Errors and faults

All validation and acceptance failures return `Err(FaultCode.COMMAND_INVALID)`
without mutating input state.

## Messages

None. `TickInputs.command` is a `RoutedCommandMsg` value; nothing here
publishes.

## Configuration

None. Policy limits arrive as explicit `PolicyLimits` values.

## Constraints

The module is pure and mode-free. There is no graph engine, no callable
dispatch, and no live runtime state here.

## Related documents

- [`flight.payload.graphs`](../graphs.md)
- [`flight.payload.records`](../records.md)
- [`flight.payload.gimbal.request`](../gimbal/request.md)
