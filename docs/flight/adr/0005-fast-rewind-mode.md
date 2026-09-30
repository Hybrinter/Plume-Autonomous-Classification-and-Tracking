# ADR-FLIGHT-0005: FAST_REWIND is an arbiter mode

**Status:** Accepted
**Date:** 2026-09-17
**Topic:** feature-add
**Supersedes:** none
**Superseded-by:** none
**Related:** ADR-FLIGHT-0004

## Context

ADR-FLIGHT-0004 defined a two-phase hunt behind one `GimbalState.REWIND` value.
`outer_rate` switched from smear-capped hunt to hardware slew when
`rewind_elapsed_s >= rewind_sharp_max_s`. The arbiter only stamped
`rewind_entered_s`. That split ownership of the policy switch.

The same smear budget already limits `TRACKING` and sharp `REWIND` in
`outer_rate`. Raising `MOTION_SMEAR` from a second copy of that inequality was
redundant and disagreed with the command limiter. Hardware-slew frames smear on
purpose.

`GimbalCommandMode` remains pose only (`ABSOLUTE` / `STOW` / `HOME`). A rate
policy does not belong on that enum.

## Decision

- Add `GimbalState.FAST_REWIND`. Loss below the limb still enters `REWIND`.
  The arbiter promotes to `FAST_REWIND` after `rewind_sharp_max_s` when there
  is no plume and the gimbal is not at the limb.
- `outer_rate` is mode-only. `REWIND` always hunts at `omega_t_nom + omega_sharp`.
  `FAST_REWIND` always commands `omega_hw`. Do not pass elapsed time into the
  rate core.
- Treat `REWIND` and `FAST_REWIND` the same for plume, limb, SAFE, boresight
  scene rate, residual freeze, and acquire reset. Keep `rewind_entered_s`
  through `FAST_REWIND`.
- Do not raise `FrameUsabilityTag.MOTION_SMEAR`. Dataset exclusion for the
  hardware-slew hunt is `gimbal_state == FAST_REWIND`. Keep
  `max_motion_smear_px` as the control smear cap.

## Consequences

- Pointing telemetry reports `gimbal_state` instead of a `rewind_escape` timer
  flag.
- A starved sharp hunt still waits for the arbiter timer before hardware slew.
- Residual `rewind_update` is unchanged. That name is a filter snapshot, not
  this hunt mode.

## Alternatives considered

- Put FAST_REWIND on `GimbalCommandMode` — that enum is HAL pose, not the
  pointing FSM.
- Keep the timer inside `outer_rate` — the arbiter would still hide two
  policies behind one state.
- Gate training on `MOTION_SMEAR` — duplicates the control cap and flags the
  one mode that must smear.
