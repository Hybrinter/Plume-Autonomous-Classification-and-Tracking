# sim.sil

**Source:** `packages/sim/src/sim/sil/`
**Kind:** package

## Purpose

The SIL package wires flight apps over sim or profile-selected drivers and steps them in one
thread. It exposes builders, harnesses, and the shared `step_once` cycle body.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`runner`](sil/runner.md) | module | All-sim `SilSystem`, `build_sil_system`, `SilHarness` |
| [`stepping`](sil/stepping.md) | module | Driver-agnostic `step_once` for one SIL cycle |
| [`validation`](sil/validation.md) | module | Env-driven `ValidationSystem`, harness, profile loader |

## Package interface

`sim.sil.__init__` re-exports:

| Name | Kind |
| --- | --- |
| `SilHarness` | class |
| `SilSystem` | class |
| `SimDriverInputs` | class (from `flight.core.select_drivers`) |
| `ValidationHarness` | class |
| `ValidationSystem` | class |
| `build_sil_system` | function |
| `build_validation_system` | function |
| `load_profile_config` | function |
| `step_once` | function |

## Interactions

SIL imports flight composition, config, HAL protocols, payload, and fault modules. It calls
`flight.core.composition.build_apps` and `flight.core.select_drivers.select_drivers`.

`EnvironmentConfig` selects sim or real drivers per axis (`sensor`, `gimbal`, `compute`,
`link`, `clock`, plus `host`). Profiles under `profiles/*.toml` override `config/default.toml`.

Command-path tests build signed telecommands with `flight.libs.commands.build_tc_packet` and
pass the bytes as `inbound_packets` to `build_sil_system`.

GSE drives `build_validation_system` and `ValidationHarness`. Tools analysis calls
`build_sil_system` and `step_once` directly.

## Constraints

- `step_once` is the single source of truth for one deterministic cycle.
- The harness publishes one `HeartbeatMsg` per entry in `MONITORED_SUBSYSTEMS` each step.
- `SilHarness.run_steps` advances the shared `ManualClock` so `SimGimbal` dynamics integrate.
- Storage redirects to a temp directory in `build_validation_system`.

## Related documents

- [`sim`](sim.md)
- [`sim.sil.runner`](sil/runner.md)
- [`sim.sil.stepping`](sil/stepping.md)
- [`sim.sil.validation`](sil/validation.md)
- [`gse`](gse.md)
