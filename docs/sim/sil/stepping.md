# sim.sil.stepping

**Source:** `packages/sim/src/sim/sil/stepping.py`
**Kind:** shell utility

## Purpose

`step_once` runs one deterministic SIL cycle over the shared bus and owns forward
advancement of the shared `ManualClock`. It is driver-agnostic and threads payload and
FDIR state in and out. Optional `SilCycleBind.pre_step` runs after loop catch-up and
before acquire.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SilCycleBind` | Protocol | `pre_step(now)` world evaluate and driver feed |
| `step_once` | function | Advance every subsystem one cycle; return new state |

## Inputs and outputs

**`step_once(apps, sensor, gimbal, bus, clock, now, payload_state, fault_entries, bind=None)`**

- Inputs: `SystemApps`, `ImagingSensor`, `GimbalActuator`, `MessageBus`, `ManualClock`,
  monotonic `now`, payload `PayloadState`, FDIR watchdog entry map, optional `SilCycleBind`.
- Output: tuple of new `PayloadState` and new watchdog entry map.

## Behavior

1. Tick the system-mode authority, then drain payload activations and safety
   evidence at the current clock (`poll_activations`). Activations are
   published only by the real `SystemModesApp` (or an explicit test-fixture
   injection); nothing here fabricates authority. The first authority tick
   boots SAFE, so the boot activation is visible before the payload's drain.
2. Catch up the control loops to `now`: seed one `control_tick` at the current
   clock, then advance the shared `ManualClock` in inner-period deadlines
   (`min(current + dt, now)`, terminating within `1e-12`), calling the same
   `control_tick` seam at each deadline. Physical device time therefore
   progresses even while the actuator is inhibited; routed payload commands
   commit inside the outer tick.
3. Take one control-owned feedback sample (`sample_feedback`) so a non-grid
   shutter has actual encoder evidence, then run `bind.pre_step(now)` when a
   bind is present.
4. Run one planned capture cycle (`capture_once`): the same pure
   `plan_capture` deadline/duty schedule used by the production `run` loop
   decides `WAIT` (no I/O), `DRAIN` (release one buffered frame), or
   `CAPTURE` (acquire and process). Repeated early harness calls spend no
   opportunities; deadline semantics are identical in SIL and production.
5. Run iss_iface and command_router ticks, then tick the authority again so a
   routed `SET_MODE`, `EXIT_SAFE`, or `GIMBAL_STOW` is decided in the same
   cycle.
6. Run thermal and electrical handle-commands and sample.
7. Run model_deploy, storage, and downlink ticks.
8. Publish one `HeartbeatMsg` per name in `MONITORED_SUBSYSTEMS`.
9. Run the fault app tick, then tick the authority once more so a fault SAFE
   request is arbitrated in the same cycle, and return updated state.

`step_once` owns forward advancement of the shared clock to `now` and never rewinds
it; callers must not advance the clock separately. Vision from a captured frame waits
for a later outer tick. A missing encoder bracket stays `theta_g_rad is None`. Ground
pose commands routed this cycle apply on a later cycle's catch-up.

## Errors and faults

A `DRAIN` cycle releases one buffered frame; it does not acquire. Sensor
acquire or gimbal read failures skip `process_frame` for that cycle, and a
current imaging failure queues containment for the next control poll through
`pending_fault`. Fault routing happens inside the fault app tick.
`bind.pre_step` may raise `ValueError` when a live mosaic would mix with
unread constructor frames.

## Messages

**Publishes:** `HeartbeatMsg` (one per monitored subsystem, `sequence=0`).

Apps publish their own types during their tick methods. The step body does not publish
processed frames or inference tensors.

## Configuration

None. Callers pass a wired `SystemApps` bundle.

## Constraints

- Imports HAL protocols and apps only. No concrete driver modules.
- Holds no module-level mutable state. State is always threaded in and out.
- `SilHarness`, `ValidationHarness`, GSE `InProcessBackend`, and the tools analysis
  recorder all delegate here. Bind invocation lives inside this cycle body.
- Only this function advances the shared clock; harnesses and backends do not
  advance it around `step` calls.

## Related documents

- [`sim.sil`](sil.md)
- [`sim.sil.runner`](runner.md)
- [`sim.sil.validation`](validation.md)
- [`sim.sil.environment_bind`](environment_bind.md)
- [`gse.harness`](gse/harness.md)
