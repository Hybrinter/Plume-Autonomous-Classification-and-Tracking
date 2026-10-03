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

1. Drain payload activations and safety evidence at the current clock
   (`poll_activations`). Activations arrive only via explicit
   `SystemModeActivatedMsg` publications; nothing fabricates authority.
2. Catch up the control loops to `now`: seed one `control_tick` at the current
   clock, then advance the shared `ManualClock` in inner-period deadlines
   (`min(current + dt, now)`, terminating within `1e-12`), calling the same
   `control_tick` seam at each deadline. Physical device time therefore
   progresses even while the actuator is inhibited; routed payload commands
   commit inside the outer tick.
3. Take one control-owned feedback sample (`sample_feedback`) so a non-grid
   shutter has actual encoder evidence, then run `bind.pre_step(now)` when a
   bind is present.
4. Run one conservative capture cycle (`capture_once`): acquire and process when
   the policy/duty gate is due, otherwise drain one unread frame.
5. Run iss_iface and command_router ticks.
6. Run thermal and electrical handle-commands and sample.
7. Run model_deploy, storage, and downlink ticks.
8. Publish one `HeartbeatMsg` per name in `MONITORED_SUBSYSTEMS`.
9. Run the fault app tick and return updated state.

`step_once` owns forward advancement of the shared clock to `now` and never rewinds
it; callers must not advance the clock separately. Vision from a captured frame waits
for a later outer tick. A missing encoder bracket stays `theta_g_rad is None`. Ground
pose commands routed this cycle apply on a later cycle's catch-up.

## Errors and faults

An off-duty cycle drains one frame; it does not acquire. Sensor acquire or gimbal
read failures skip `process_frame` for that cycle. Fault routing happens inside the
fault app tick. `bind.pre_step` may raise `ValueError` when a live mosaic would mix
with unread constructor frames.

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
