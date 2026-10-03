# flight.payload.imaging

**Source:** `packages/flight/src/flight/payload/imaging.py`
**Kind:** pure module

## Purpose

The module owns the capture loop's timing decisions as pure functions so the
production `run` loop and the SIL `step_once` seam share identical cadence.
It plans deadline, duty-floor, and inference-decimation decisions; the
imperative app shell executes them.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `CaptureDecision` | enum | `WAIT`, `DRAIN`, or `CAPTURE` for one call |
| `CaptureSchedule` | dataclass | Context, next deadline, opportunity and capture counters |
| `CapturePlan` | dataclass | One decision plus the schedule that produced it |
| `plan_capture` | function | `Result[CapturePlan, FaultCode]` per capture-loop call |
| `record_capture` | function | Counts one successful capture and gates inference |

## Inputs and outputs

`plan_capture(schedule, policy, context, now, limits)` takes the previous
schedule, the resolved `EffectivePolicy`, the shell-stamped `CaptureContext`,
monotonic `now`, and the sensor `PolicyLimits`. It returns `Err` for a
nonfinite time or an invalid complete policy, otherwise `Ok` with `WAIT`,
`DRAIN`, or `CAPTURE` plus the updated schedule.

`record_capture(schedule, context, inference)` returns the updated schedule
and whether inference runs on this successful capture.

## Behavior

1. A changed context (activation key, policy revision, or containment
   generation) resets phase and counters; the first opportunity is due at the
   first call.
2. Calls before `next_opportunity_s` return `WAIT` and spend nothing. A
   validated policy with acquisition disabled also plans `WAIT`: nothing is
   spent and no deadline advances. A due call spends exactly one opportunity
   and sets the next deadline at `now + capture_interval_s`; a late call
   never bursts missed frames.
3. The duty floor `floor(index * duty) > floor((index - 1) * duty)`, index
   starting at 1, selects `CAPTURE`; other due opportunities `DRAIN`. Duty
   0.5 captures even opportunities.
4. `record_capture` increments only successful captures. Enabled inference
   runs on the first successful capture and then every `every_n_frames`-th:
   `(captured_frames - 1) % N == 0`. A skipped frame runs no detector, emits
   no vision, and is not a plume-loss observation.

## Errors and faults

`Err(COMMAND_INVALID)` for nonfinite `now` or an invalid resolved policy.
`record_capture` cannot fail; failed acquisitions never reach it.

## Messages

None.

## Configuration

Reads `EffectivePolicy` and `PolicyLimits` values; it never reads
`PactConfig` or sensor fields directly.

## Constraints

Pure: no I/O, no bus, no clock reads, no logging. The shell calls `drain_frame`
only on `DRAIN` and `acquire_frame` only on `CAPTURE`.

## Related documents

- [`flight.payload.app`](app.md)
- [`flight.payload.graphs.base`](graphs/base.md)
- [`flight.payload.records`](records.md)
