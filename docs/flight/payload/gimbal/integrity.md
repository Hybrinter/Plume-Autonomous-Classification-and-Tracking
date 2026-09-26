# flight.payload.gimbal.integrity

**Source:** `packages/flight/src/flight/payload/gimbal/integrity.py`
**Kind:** pure module

## Purpose

The light pointing-integrity detector trips `GIMBAL_RUNAWAY` on NaN torque or rate
or an encoder freeze under a nonzero rate reference. It is not the deleted
RATE-mode runaway monitor.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `IntegrityResult` | dataclass | Updated strike counts and optional fault |
| `check_integrity` | function | Advance strikes and trip when a rule fires |

## Inputs and outputs

`check_integrity` takes `IntegrityConfig`, the rate reference, encoder-rate
estimate, torque, two-sample encoder rate, and prior strike counts. It returns
an `IntegrityResult`.

## Behavior

1. A non-finite `r`, `y_m`, or `tau` trips immediately.
2. Encoder freeze: `|r|` above `r_min_rad_s` and `|encoder_rate|` below
   `encoder_rate_ratio * |r|` for `freeze_strikes` inner ticks.
3. A healthy sample resets the matching counter.

## Errors and faults

| Fault | Trigger |
| --- | --- |
| `GIMBAL_RUNAWAY` | NaN or freeze strikes |

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
