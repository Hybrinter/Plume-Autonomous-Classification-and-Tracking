# `gse` Subsystem Context

Non-obvious context for the ground-support-equipment package (`pact-gse`). The GSE stands in for
the real ground segment in validation; it is **test tooling**, not flight software.

---

## One-way leaf in the layer graph

`gse` may import `flight.libs` + `sim` ONLY. `flight` and `sim` must **never** import `gse`
(enforced by the `flight-gse-isolation` and `sim-gse-isolation` import-linter contracts). The
emulator is a consumer of the flight wire formats, never a dependency of the flight build.

## Builds packets from the canonical command codec

`StationEmulator.send_command` builds signed CCSDS telecommands with
`flight.libs.commands.build_tc_packet` -- the canonical home of that symbol after Phase A relocated
it there (the `flight.iss_iface.ingress` path is a back-compat re-export only). Import it from
`flight.libs.commands`, never from `flight.iss_iface.ingress`.

## The validation config matrix

The validation venues are selected by `flight.libs.config.config.EnvironmentConfig` (five
`AxisMode` axes -- `sensor`/`gimbal`/`compute`/`link`/`clock` -- plus `host`), applied as
`profiles/*.toml` overrides. Running venues: `sil` (all sim) and `sil-link-real` (link real).
DEFINED-NOT-RUN venues: `pil`, `hil` (see `docs/validation/`). `gse.orchestrator.run_scenario`
scores **frame-portable** assertions and records **realtime-only** assertions as `skip` under the
deterministic in-process backend.

## Shared stepper -- reuse, do not fork

`gse.harness.InProcessBackend` steps the flight apps via `sim.sil.stepping.step_once(...)` -- the
exact same driver-agnostic function `SilHarness.step` delegates to. The in-process backend builds
config via `load_config(default, profile)`, picks `ManualClock`, builds `SimDriverInputs` from the
scene, calls `select_drivers` + `build_apps`, and (for `link="real"`) stands up a
`StationEmulator`. `SocketBackend` (PIL/HIL) is deferred and raises `NotImplementedError`.

## Permanent gap: the real ground segment is never tested

`StationEmulator` is a stand-in. The **real ground segment** is never exercised by any running or
defined venue -- this is a permanent VCRM gap recorded in `docs/requirements/vcrm.toml`
(`GAP-GROUND-SEGMENT`). The `lock` (LaunchLock) axis is a second permanent gap: no device, no
config field.
