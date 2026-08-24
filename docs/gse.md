# gse

**Source:** `packages/gse/src/gse/`
**Kind:** package

## Purpose

GSE is out-of-flight test tooling. It emulates the ground segment and runs declarative
scenarios against the flight software through a transport-agnostic harness backend.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`harness`](gse/harness.md) | module | `InProcessBackend`, `SocketBackend`, `TelemetryCapture` |
| [`orchestrator`](gse/orchestrator.md) | module | Scenario runner, scoring, V&V evidence export |
| [`scenario`](gse/scenario.md) | module | Frozen scenario model and TOML loader |
| [`station`](gse/station.md) | module | TCP/UDP `StationEmulator` for real link profiles |

## Package interface

`gse.__init__` carries module docstring only. Import from `gse.harness`, `gse.orchestrator`,
`gse.scenario`, or `gse.station`.

## Interactions

GSE imports `flight.libs` (commands, config, messages, types) and `sim` (scene, validation
harness, `step_once` via the in-process backend). Flight and sim never import gse.

The in-process backend steps flight apps through `sim.sil.ValidationHarness`, which delegates
to `step_once`. Real-link profiles stand up `StationEmulator` as the ground-segment
counterpart to `RealStationLink`.

## Constraints

- One-way dependency: gse may import flight.libs and sim only.
- `StationEmulator` is a test stand-in. The real ground segment is not exercised by any
  running venue.
- `SocketBackend` is a stub. Every method raises `NotImplementedError`.
- Sim-link scenarios pre-bake commands at build time. `at_frame` timing applies on real link
  only.

## Related documents

- [`gse.harness`](gse/harness.md)
- [`gse.orchestrator`](gse/orchestrator.md)
- [`gse.scenario`](gse/scenario.md)
- [`gse.station`](gse/station.md)
- [`sim.sil`](sim/sil.md)
