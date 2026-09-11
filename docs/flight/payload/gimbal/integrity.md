# flight.payload.gimbal.integrity

**Source:** `packages/flight/src/flight/payload/gimbal/integrity.py`
**Kind:** pure module

## Purpose

The light pointing-integrity detector trips `GIMBAL_RUNAWAY` on NaN torque or rate,
an encoder freeze under a nonzero rate reference, or motion against an engaged
launch lock. It is not the deleted RATE-mode runaway monitor.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `IntegrityResult` | dataclass | Updated strike counts and optional fault |
| `check_integrity` | function | Advance strikes and trip when a rule fires |
| `lock_hold_rate` | function | Hold-pose motion since lock engage |

## Inputs and outputs

`check_integrity` takes `IntegrityConfig`, the rate reference, encoder-rate
estimate, torque, two-sample encoder rate, lock flag, prior strike counts, and
hold-pose motion since lock engage. It returns an `IntegrityResult`.

`lock_hold_rate` takes the lock flag, current encoder angle, now, and the latched
pose. It returns motion in rad/s and the updated pose reference.

## Behavior

1. A non-finite `r`, `y_m`, `tau`, or `lock_motion_rad_s` trips immediately.
2. Encoder freeze: `|r|` above `r_min_rad_s` and `|encoder_rate|` below
   `encoder_rate_ratio * |r|` for `freeze_strikes` inner ticks.
3. Lock-fight: lock engaged and `|lock_motion_rad_s|` above `lock_fight_rad_s` for
   `lock_fight_strikes` inner ticks. `lock_motion_rad_s` is hold-pose motion from
   `lock_hold_rate`, not the 1 kHz polynomial `y_m`.
4. A healthy sample resets the matching counter.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `GIMBAL_RUNAWAY` | NaN, freeze strikes, or lock-fight strikes |

The existing FDIR policy routes `GIMBAL_RUNAWAY` to SAFE.

## Messages

None. The app shell publishes `FaultEventMsg`.

## Configuration

`IntegrityConfig` under `[controller.integrity]`.

## Constraints

The functions are pure. The app shell calls them from `advance_inner` and the inner thread.

## Related documents

- [`flight.payload.app`](../app.md)
- [`flight.payload.gimbal.inner`](inner.md)
- [`flight.fault.policy`](../../fault/policy.md)
