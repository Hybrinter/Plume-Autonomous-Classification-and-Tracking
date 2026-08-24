# sim.sil.validation

**Source:** `packages/sim/src/sim/sil/validation.py`
**Kind:** module

## Purpose

The validation module builds a flight system for any environment profile and steps it
deterministically. GSE imports this surface instead of touching flight composition directly.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ValidationSystem` | class | Wired apps, bus, clock, and protocol-typed drivers |
| `build_validation_system` | function | Env-driven driver selection and app wiring |
| `ValidationHarness` | class | Single-threaded stepper over a `ValidationSystem` |
| `load_profile_config` | function | Load base TOML merged with a profile override |

## Inputs and outputs

**`build_validation_system(config, clock, sim_inputs=None, uplink_key=...) -> ValidationSystem`**

- Inputs: `PactConfig` (environment axes intact), `ManualClock`, optional `SimDriverInputs`,
  uplink HMAC key.
- Output: `ValidationSystem` with HAL protocol-typed driver fields.

**`ValidationHarness.step(now) -> None`**

- Same contract as `SilHarness.step`. Delegates to `step_once`.

**`load_profile_config(config_path, override_path) -> PactConfig`**

- Inputs: base TOML path, profile override path.
- Output: merged validated `PactConfig`.
- Raises `ValueError` when `load_config` returns `Err`.

## Behavior

1. `build_validation_system` redirects storage to a fresh temp directory.
2. It creates a new `MessageBus` and calls `select_drivers` with the supplied config.
3. It builds identity mosaic calibration from sensor dimensions.
4. It wires every app via `build_apps` with `MONITORED_SUBSYSTEMS`.
5. `ValidationHarness` seeds payload and fault state, then steps like `SilHarness`.
6. `load_profile_config` calls `flight.core.config_loader.load_config` and raises on failure.

## Errors and faults

`load_profile_config` raises `ValueError` on config load failure. Runtime faults publish on
the bus during stepping.

## Messages

Same as [`sim.sil.stepping`](stepping.md). Heartbeats are published inside `step_once`.

## Configuration

Reads the full `PactConfig`. Environment axes (`sensor`, `gimbal`, `compute`, `link`,
`clock`, `host`) drive driver selection.

Default uplink key is `b"sil-test-key-0000000000000000000"`.

## Constraints

- Driver fields stay protocol-typed. No cast is required after `select_drivers`.
- A `"real"` link axis yields `RealStationLink`. Other axes may stay sim.
- GSE is the primary consumer of this module.

## Related documents

- [`sim.sil`](sil.md)
- [`sim.sil.runner`](runner.md)
- [`sim.sil.stepping`](stepping.md)
- [`gse.harness`](gse/harness.md)
