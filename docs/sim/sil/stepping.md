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

**`step_once(apps, sensor, gimbal, bus, clock, now, payload_state, fault_entries, dt=1.0)`**

- Inputs: `SystemApps`, `ImagingSensor`, `GimbalActuator`, `MessageBus`, `ManualClock`,
  monotonic `now`, payload `ControlState`, FDIR watchdog entry map, scenario `dt`.
- Output: tuple of new `ControlState` and new watchdog entry map.

## Behavior

1. Poll payload mode changes and launch-lock state.
2. Acquire one frame from the sensor. On success, call `process_frame` (vision ingest).
3. Split `dt` into inner chunks of `dt_inner_max_s`. Advance the clock each chunk and
   call `apply_control` (outer LQG plus inner PI plus `set_torque` or stow). SAFE flags
   apply on the first chunk only. Skip this loop when `dt` is not positive.
4. Run iss_iface, command_router, and mechanical ticks.
5. Run thermal and electrical handle-commands and sample.
6. Run model_deploy, storage, and downlink ticks.
7. Publish one `HeartbeatMsg` per name in `MONITORED_SUBSYSTEMS`.
8. Run the fault app tick and return updated state.

Ingress, routing, and command execution occur in the same cycle. Downlink items emitted this
cycle transmit on the next iss_iface tick.

## Errors and faults

Sensor acquire or gimbal read failures skip `process_frame` for that cycle. Control ticks
still run. Fault routing happens inside the fault app tick.

## Messages

**Publishes:** `HeartbeatMsg` (one per monitored subsystem, `sequence=0`).

Apps publish their own types during their tick methods. The step body does not publish
processed frames or inference tensors.

## Configuration

Reads inner/outer Δt bounds from the payload `ControllerConfig` via `apps.payload`.

## Constraints

- Imports HAL protocols and apps only. No concrete driver modules.
- Holds no module-level mutable state. State is always threaded in and out.
- `SilHarness`, `ValidationHarness`, GSE `InProcessBackend`, and tools recorder all delegate
  here. Those callers do not pre-advance the clock; `step_once` advances it per inner chunk.

## Related documents

- [`sim.sil`](sil.md)
- [`sim.sil.runner`](runner.md)
- [`sim.sil.validation`](validation.md)
- [`gse.harness`](gse/harness.md)
