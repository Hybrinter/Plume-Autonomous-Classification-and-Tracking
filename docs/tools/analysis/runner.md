# tools.analysis.runner

**Source:** `packages/tools/src/tools/analysis/runner.py`
**Kind:** module

## Purpose

The runner defines declarative SIL scenario specs, wires `SilSystem` instances, and captures
runs through the passive recorder. It covers nominal tracking and fault or command paths.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `Injection` | class | Timed bus message to publish before a step |
| `Action` | class | Timed callable on the wired system before a step |
| `ScenarioSpec` | class | Full declarative SIL run description |
| `ScenarioRun` | class | Spec plus recorder `CaptureResult` |
| `SCENARIOS` | constant | Built-in scenario registry keyed by name |
| `build_system` | function | Wire a fresh `SilSystem` for a spec |
| `run_scenario` | function | Capture one scenario end-to-end |
| `scenario` | function | Lookup built-in spec by name |
| `scenario_names` | function | All built-in names in declaration order |
| `load_scenario_spec` | function | Adapt a GSE-style scenario TOML to `ScenarioSpec` |

## Inputs and outputs

**`build_system(spec) -> SilSystem`**

- Calls `build_sil_system` with plume frames, detector, readings, packets, and config from
  the spec.

**`run_scenario(spec) -> ScenarioRun`**

- Output: spec plus `record_run` capture with optional pre-step hook for injections and
  actions.

**`load_scenario_spec(path) -> ScenarioSpec`**

- Converts GSE TOML commands to post-ingress `CommandMsg` injections via the command
  dictionary. Ignores GSE assertions. Extra GSE keys are ignored by the file schema.

## Behavior

1. Built-in scenarios set steps, frame counts, thermal or power scripts, injections, actions,
   inbound `build_tc_packet` bytes, or shrunk storage or downlink quotas.
2. `_make_pre_step` groups actions and injections by 1-based step index.
3. Pre-step runs actions first, then publishes injection messages on the bus.
4. `record_run` owns the stepping loop after the hook fires.

Built-in scenarios include: nominal tracking, thermal and power SAFE, gimbal runaway,
watchdog inject, EXIT_SAFE recovery, hazardous ARM/EXECUTE, launch-lock interlock, model
lifecycle, storage eviction, downlink AOS budget, and signed command ingress.

## Errors and faults

`scenario` raises `KeyError` for unknown names. TOML load raises on malformed files.

## Messages

Injections publish types such as `CommandMsg`, `FaultEventMsg`, and `ModelStagedMsg`. Actions
may call public storage APIs or flip sim link state.

## Configuration

Each `ScenarioSpec` carries an optional `PactConfig`. Helpers shrink storage quota or
downlink budget for limit exercises.

Default uplink key is `b"sil-test-key-0000000000000000000"`.

## Constraints

- Faults the harness cannot raise organically (runaway, watchdog miss) arrive as injected
  `FaultEventMsg`.
- Gimbal runaway is injected. The sim gimbal tracks commands faithfully.
- File scenarios prefix names with `file_` and category `scenario-file`.

## Related documents

- [`tools.analysis`](analysis.md)
- [`tools.analysis.recorder`](recorder.md)
- [`tools.analysis.characterize`](characterize.md)
- [`sim.sil.runner`](sim/sil/runner.md)
