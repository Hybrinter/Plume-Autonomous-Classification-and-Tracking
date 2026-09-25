# sim.sil.stepping

**Source:** `packages/sim/src/sim/sil/stepping.py`
**Kind:** pure module

## Purpose

`step_once` runs one deterministic SIL cycle over the shared bus. It is driver-agnostic and
threads payload and FDIR state in and out. Optional `SilCycleBind.pre_step` runs after loop
catch-up and before acquire.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SilCycleBind` | Protocol | `pre_step(now)` world evaluate and driver feed |
| `step_once` | function | Advance every subsystem one cycle; return new state |

## Inputs and outputs

**`step_once(apps, sensor, gimbal, bus, clock, now, payload_state, fault_entries, bind=None)`**

- Inputs: `SystemApps`, `ImagingSensor`, `GimbalActuator`, `MessageBus`, `ManualClock`,
  monotonic `now`, payload `ControlState`, FDIR watchdog entry map, optional `SilCycleBind`.
- Output: tuple of new `ControlState` and new watchdog entry map.

## Behavior

1. Publish the current launch-lock driver state so fail-closed payload sees it
   on step 1. Poll mode changes and lock.
2. Catch up inner and outer loops to `now`. For each `T_out` slice: `advance_inner`
   up to tick `t`, then one `outer_step` at `t`. Then trailing inner to `now`. A first
   step with `last_outer_s is None` stamps origins, runs inner through `now`,
   then outer, then trailing inner. Rate-mode inner records one encoder sample
   per call and does not run the detailed-plant PI. Bound-frame vision still
   waits until after this catch-up.
3. When a bind is present, call `bind.pre_step(now)`. Catch-up has already integrated
   the plant through shutter. `advance_plant` then sees a frozen clock and leaves
   catch-up debt in place.
4. Acquire one frame from the sensor. On success, read gimbal position and call
   `process_frame` with that position. Encoder samples from catch-up and this read
   share shutter time `now` with the frame stamp.
5. Apply payload pose commands from the prior cycle (`handle_commands`).
6. Run iss_iface, command_router, and mechanical ticks.
7. Run thermal and electrical handle-commands and sample.
8. Run model_deploy, storage, and downlink ticks.
9. Publish one `HeartbeatMsg` per name in `MONITORED_SUBSYSTEMS`.
10. Run the fault app tick and return updated state.

Ingress, routing, and command execution occur in the same cycle. Downlink items emitted this
cycle transmit on the next iss_iface tick. Vision from this frame waits for a later outer
tick. A missing encoder bracket stays `theta_g_rad is None`. Ground pose commands routed
this cycle apply on a later cycle's catch-up.

## Errors and faults

`capture_this_opportunity` skips acquire on off-duty cycles and still records gimbal
feedback. Sensor acquire or gimbal read failures skip `process_frame` for that cycle. Fault routing
happens inside the fault app tick. `bind.pre_step` may raise `ValueError` when a live mosaic
would mix with unread constructor frames.

## Messages

**Publishes:** `HeartbeatMsg` (one per monitored subsystem, `sequence=0`).

Apps publish their own types during their tick methods. The step body does not publish
processed frames or inference tensors.

## Configuration

None. Callers pass a wired `SystemApps` bundle.

## Constraints

- Imports HAL protocols and apps only. No concrete driver modules.
- Holds no module-level mutable state. State is always threaded in and out.
- `SilHarness`, `ValidationHarness`, GSE `InProcessBackend`, and tools recorder all delegate
  here. Bind invocation lives inside this cycle body.

## Related documents

- [`sim.sil`](sil.md)
- [`sim.sil.runner`](runner.md)
- [`sim.sil.validation`](validation.md)
- [`sim.sil.environment_bind`](environment_bind.md)
- [`gse.harness`](gse/harness.md)
