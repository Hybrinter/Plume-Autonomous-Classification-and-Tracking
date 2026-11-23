# ADR-FLIGHT-0008: Separate system-mode request, notification, and activation messages

**Status:** Superseded
**Date:** 2026-10-02
**Topic:** interface
**Supersedes:** ADR-REPO-0006 (SAFE transition contract), ADR-REPO-0008 (SAFE transition contract), ADR-REPO-0009 (EXIT_SAFE execution acknowledgements)
**Superseded-by:** ADR-FLIGHT-0009
**Related:** ADR-FLIGHT-0007

## Context

Today `ModeChangeMsg` is dual-purpose: subsystems publish it to request SAFE and
consumers treat it as the mode that took effect. Commands such as `EXIT_SAFE`
are executed and ACKed by the fault app directly. With a teammate-owned mode
authority, request, recorded outcome, and behavior-changing activation are
different facts and must not share one message.

## Decision

- Define distinct frozen contract messages: `SystemModeRequestMsg` (request
  identity, requested mode, requester, reason, optional correlation),
  `SystemModeTransitionMsg` (transition identity, epoch, previous/requested/
  resulting mode, ACCEPTED/DENIED, reason; audit/storage/downlink only), and
  `SystemModeActivatedMsg` (epoch, sequence, previous mode, active mode, reason,
  optional request identity, `recovery_authorized`).
- Add `SystemModeSyncRequestMsg` so a restarting subscriber can request the
  current activation snapshot; the bus retains no publications for late
  subscribers.
- Only a validated activation changes payload behavior. Requests and
  notifications -- accepted or denied -- never select a graph or release motion.
- Activation identity is `(epoch, sequence)`: the epoch is created once by the
  composition root; the sequence is owned by the teammate's authority. Payload
  rejects unexpected epochs, ignores older sequences, and treats conflicting
  contents under one key as a synchronization error.
- `recovery_authorized` defaults false and may be true only for an
  authority-approved `EXIT_SAFE`; ordinary activations never clear fault-owned
  or hardware latches.
- Major command targets are `"system_modes"`: `GIMBAL_STOW` requests `STOW` and
  `EXIT_SAFE` requests exit to `IDLE`. The authority owns those execution ACKs;
  the `"core"` target is not used because the router executes core commands
  directly.
- At cutover, remove `ModeChangeMsg`, `MessageType.MODE_CHANGE`, the old
  system-mode members, and `SafetyStateMsg.mode`; the fault latch is safety
  evidence, not the active mode. Incompatible changes increment the global
  schema version.
- Runtime implementation is pending. Teammate agreement on these fields,
  sequence/epoch ownership, the snapshot response, recovery authorization, ACK
  ownership, and boot synchronization is a pending external dependency; this
  record freezes the contract this stack will implement against.

## Consequences

- Consumers correlate transitions and activations by identity, not by assumed
  cross-subscription queue order. Current behavior is reconstructable from the
  activation alone.
- Storage/downlink persist the dedicated transition record through explicit
  consumers; no duplicate generic event becomes a second canonical record.
- Fault containment works even when a SAFE request is denied or delayed: the
  latch inhibits immediately on safety evidence independent of activation.

## Alternatives considered

- Keep `ModeChangeMsg` as both request and record -- conflates denied requests
  with committed behavior and hides denial evidence.
- Let subscribers infer the active mode from notification history -- fails late
  subscribers and restart synchronization.
