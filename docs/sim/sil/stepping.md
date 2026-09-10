# sim.sil.stepping

**Source:** `packages/sim/src/sim/sil/stepping.py`
**Kind:** pure module

## Purpose

`step_once` runs one deterministic SIL cycle over the shared bus. It is driver-agnostic and
threads payload and FDIR state in and out.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `step_once` | function | Advance every subsystem one cycle; return new state |

## Inputs and outputs

**`step_once(apps, sensor, gimbal, bus, clock, now, payload_state, fault_entries)`**

- Inputs: `SystemApps`, `ImagingSensor`, `GimbalActuator`, `MessageBus`, `ManualClock`,
  monotonic `now`, payload `ControlState`, FDIR watchdog entry map.
- Output: tuple of new `ControlState` and new watchdog entry map.

## Behavior

1. Publish the current launch-lock driver state so fail-closed payload sees it
   on step 1. Poll mode changes and lock. Apply payload pose commands from the
   prior cycle.
2. Acquire one frame from the sensor. On success, read gimbal position and call
   `process_frame`.
3. For each `T_out` slice: `advance_inner` up to tick `t`, then one `outer_step`
   at `t`. Then trailing inner to `now`. A first step with `last_outer_s is None`
   runs outer then inner once so origins stamp.
4. Run iss_iface, command_router, and mechanical ticks.
5. Run thermal and electrical handle-commands and sample.
6. Run model_deploy, storage, and downlink ticks.
7. Publish one `HeartbeatMsg` per name in `MONITORED_SUBSYSTEMS`.
8. Run the fault app tick and return updated state.

Ingress, routing, and command execution occur in the same cycle. Downlink items emitted this
cycle transmit on the next iss_iface tick.

## Errors and faults

Sensor acquire or gimbal read failures skip `process_frame` for that cycle. Fault routing
happens inside the fault app tick.

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
  here.

## Related documents

- [`sim.sil`](sil.md)
- [`sim.sil.runner`](runner.md)
- [`sim.sil.validation`](validation.md)
- [`gse.harness`](gse/harness.md)
