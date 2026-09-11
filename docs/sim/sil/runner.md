# sim.sil.runner

**Source:** `packages/sim/src/sim/sil/runner.py`
**Kind:** module

## Purpose

The SIL runner builds an all-sim flight system and drives it with a single-threaded harness.
It casts concrete sim drivers back from the validation builder for test inspection.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SilSystem` | class | Wired apps, bus, clock, and concrete sim drivers |
| `build_sil_system` | function | Force all-sim env and delegate to `build_validation_system` |
| `SilHarness` | class | Deterministic stepper over a `SilSystem` |

## Inputs and outputs

**`build_sil_system(config, clock, frames, detector, ...) -> SilSystem`**

- Inputs: `PactConfig`, `ManualClock`, mosaic frame list, `ScriptedDetector`, optional
  inbound CCSDS packets, thermal and power reading scripts, uplink HMAC key, launch-lock
  engaged flag.
- Output: frozen `SilSystem` with concrete `SimSensor`, `SimGimbal`, `SimStationLink`, and
  scalar sensors.

**`SilHarness.step(now) -> None`**

- Input: monotonic seconds for arbiter and watchdog.
- Side effect: advances one cycle via `step_once`; updates threaded payload and fault state.

**`SilHarness.run_steps(count, dt=1.0) -> None`**

- Inputs: step count, seconds per step.
- Side effect: advances clock and `now`, then calls `step` each iteration.

**`SilHarness.payload_gimbal_state() -> GimbalState`**

- Output: current arbiter gimbal state (test accessor).

## Behavior

1. `build_sil_system` packs sim inputs into `SimDriverInputs`.
2. It replaces `config.environment` with all `"sim"` axes and host `"x86_64"`.
3. It calls `build_validation_system` and casts driver fields to concrete sim types.
4. `SilHarness.__init__` seeds payload `ControlState` and FDIR watchdog entries.
5. `SilHarness.step` delegates to `step_once` with apps, protocols, bus, clock, and state.
6. `run_steps` continues from the last `now`, adds `dt` each step, and advances the
   shared clock. A later `run_steps` call does not reset time.

## Errors and faults

None at the library level. Driver and app faults surface on the bus during stepping.

## Messages

The harness does not publish directly. `step_once` publishes heartbeats and apps publish
their own message types.

## Configuration

Reads the supplied `PactConfig`. Overrides `environment` to all-sim inside
`build_sil_system`.

Default uplink key is `b"sil-test-key-0000000000000000000"`.

## Constraints

- Uses the same env-driven selection and wiring path as flight and GSE.
- No scheduler threads run. Each step calls app methods directly.
- Default uplink key must match keys used in `build_tc_packet` for command-path tests.

## Related documents

- [`sim.sil`](sil.md)
- [`sim.sil.stepping`](stepping.md)
- [`sim.sil.validation`](validation.md)
