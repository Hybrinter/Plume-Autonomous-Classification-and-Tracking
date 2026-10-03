# sim.sil.validation

**Source:** `packages/sim/src/sim/sil/validation.py`
**Kind:** module

## Purpose

The validation module builds a flight system for any driver profile and steps it
deterministically. GSE imports this surface and does not touch flight composition directly.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ValidationSystem` | class | Wired apps, bus, clock, and protocol-typed drivers |
| `build_validation_system` | function | Env-driven driver selection and app wiring |
| `ValidationHarness` | class | Single-threaded stepper over a `ValidationSystem` |
| `ActivationTarget` | Protocol | Read-only structural holder (apps, bus, clock) for activation injection |
| `publish_activation` | function | Publish one explicit `SystemModeActivatedMsg` test seam |
| `load_profile_config` | function | Load base TOML merged with a profile override |

## Inputs and outputs

**`build_validation_system(config, clock, sim_inputs=None, uplink_key=..., activation_epoch="sil") -> ValidationSystem`**

- Inputs: `PactConfig` (driver axes intact), `ManualClock`, optional `SimDriverInputs`,
  uplink HMAC key, the authority epoch forwarded to `build_apps` (default test
  epoch `"sil"`), and an optional keyword-only `InitializationVerifier` forwarded
  to the payload app (None keeps the pending-by-default production verifier).
- Output: `ValidationSystem` with HAL protocol-typed driver fields.

**`ValidationHarness.step(now) -> None`**

- Same contract as `SilHarness.step`. Delegates to `step_once` with the optional bind.

**`ValidationHarness.payload_graph() / payload_node() / payload_system_mode()`**

- Output: active `GraphId` value, graph node value, or the `SystemMode` of the
  last accepted activation; `None` before activation. The mode accessor reads
  the accepted activation only - never a fault latch or a mode request.

**`publish_activation(system, mode, sequence=1, ...) -> None`**

- Inputs: an `ActivationTarget` (SilSystem or ValidationSystem), the activated
  `SystemMode`, authority sequence, optional previous mode, request id, and
  recovery flag.
- Side effect: publishes one `SystemModeActivatedMsg` stamped with the system's
  configured epoch. Explicit test injection seam only; the harness runs no
  synthetic authority.

**`load_profile_config(config_path, override_path) -> PactConfig`**

- Inputs: base TOML path, profile override path.
- Output: merged validated `PactConfig`.
- Raises `ValueError` when `load_config` returns `Err`.

## Behavior

1. `build_validation_system` redirects storage to a fresh temp directory.
2. It creates a new `MessageBus` and calls `select_drivers` with the supplied config.
3. It builds identity mosaic calibration from sensor dimensions.
4. It wires every app via `build_apps` with `MONITORED_SUBSYSTEMS`.
5. `ValidationHarness` seeds `_now` from the shared clock plus the payload
   `PayloadState` and FDIR watchdog entries, then steps like `SilHarness`.
   `step_once` owns clock advancement; the harness never advances it separately.
6. `load_profile_config` calls `flight.core.config_loader.load_config` and raises on failure.

## Errors and faults

`load_profile_config` raises `ValueError` on config load failure. Runtime faults publish on
the bus during stepping.

## Messages

Same as [`sim.sil.stepping`](stepping.md). Heartbeats are published inside `step_once`.

## Configuration

Reads the full `PactConfig`. Driver axes (`sensor`, `gimbal`, `compute`, `link`,
`clock`, `host`) drive driver selection.

Default uplink key is `b"sil-test-key-0000000000000000000"`.

## Constraints

- Driver fields stay protocol-typed. No cast is required after `select_drivers`.
- A `"real"` link axis yields `RealStationLink`. Other axes may stay sim.
- GSE is the primary consumer of this module.
- The harness does not call `bind.pre_step` itself.
- `publish_activation` is the only activation path; no fabricated authority.

## Related documents

- [`sim.sil`](sil.md)
- [`sim.sil.runner`](runner.md)
- [`sim.sil.environment_bind`](environment_bind.md)
- [`sim.sil.stepping`](stepping.md)
- [`gse.harness`](gse/harness.md)
