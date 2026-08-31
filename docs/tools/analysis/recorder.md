# tools.analysis.recorder

**Source:** `packages/tools/src/tools/analysis/recorder.py`
**Kind:** module

## Purpose

The recorder owns the passive SIL stepping loop. It tabulates every registered signal each
step into tidy long and per-group wide pandas frames.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `CaptureResult` | class | Long frame, wide frames, step and column counts |
| `sample_devices` | function | One-shot read of sim HAL drivers for a step |
| `record_run` | function | Run capture loop and return `CaptureResult` |

## Inputs and outputs

**`record_run(system, steps, dt=1.0, pre_step=None) -> CaptureResult`**

- Inputs: wired `SilSystem`, positive step count, seconds per step, optional
  `PreStepHook(system, step)`.
- Output: `CaptureResult` with long and wide DataFrames.
- Raises `ValueError` when `steps <= 0`.

**`sample_devices(system) -> DeviceSample`**

- Output: gimbal measured and truth pose, rates, mode, stow switch, launch-lock state, link
  state, station send count, and replay cursors.

## Behavior

1. Subscribe to all nineteen message types before step 1.
2. Seed payload `ControlState` and FDIR watchdog entries.
3. Each step: advance `now`, run optional `pre_step`, call `step_once` (clock advances
   in inner chunks).
4. Drain passive subscriptions, sample devices once, build `SampleContext`.
5. Evaluate every signal in `REGISTRY`. Extractor exceptions become NaN or "".
6. Build master wide frame, add `.cumulative` columns for event-rate signals, split by group,
   and reshape to long format.

## Errors and faults

Extractor failures are swallowed per signal. The loop does not abort.

## Messages

Subscribes passively to all types in `MESSAGE_TYPES` from datapoints. Fan-out queues do not
steal messages from apps.

## Configuration

None.

## Constraints

- Mirrors `SilHarness.run_steps` timing (`now`, then `step_once`).
- Owns threaded payload and fault state for observability.
- Reads private sim driver fields read-only for truth pose and replay cursors.
- Never mutates flight state beyond what `step_once` and the optional hook do.

## Related documents

- [`tools.analysis`](analysis.md)
- [`tools.analysis.datapoints`](datapoints.md)
- [`tools.analysis.runner`](runner.md)
- [`sim.sil.stepping`](sim/sil/stepping.md)
