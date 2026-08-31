# gse.harness

**Source:** `packages/gse/src/gse/harness.py`
**Kind:** module

## Purpose

The harness module defines transport-agnostic scenario backends. `InProcessBackend` drives
the validation harness in one process. `SocketBackend` is declared but not implemented.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `TelemetryCapture` | class | Frozen scored bus events and downlink bytes |
| `HarnessBackend` | Protocol | Build, step, inject, collect, shutdown contract |
| `InProcessBackend` | class | ManualClock + `ValidationHarness` backend |
| `SocketBackend` | class | Stub PIL/HIL socket backend |

## Inputs and outputs

**`InProcessBackend.build(scenario, profile_path) -> None`**

- Inputs: frozen `Scenario`, path to profile TOML override.
- Side effect: loads config, renders plume scene, wires `ValidationSystem`, creates bus
  subscriptions.

**`InProcessBackend.step(now) -> None`**

- Input: monotonic seconds.
- Side effect: calls `ValidationHarness.step(now, dt)` which advances the clock in inner
  chunks via `step_once`.

**`InProcessBackend.inject_command(step) -> None`**

- Input: `CommandStep`.
- Real link: sends command through `StationEmulator`. Sim link: no-op (pre-baked packets).

**`InProcessBackend.collect() -> TelemetryCapture`**

- Output: inference count, gimbal-moved flag, mode changes, ack statuses, downlink packets.

**`SocketBackend.*`**

- Every method raises `NotImplementedError("PIL/HIL socket backend deferred")`.

## Behavior

1. `build` calls `load_profile_config("config/default.toml", profile_path)`.
2. It renders frames with `build_frames` and `plume_detector`.
3. For `link="real"`, it reserves a free TCP/UDP port pair, patches `LinkConfig`, builds the
   system, and connects `StationEmulator`.
4. For sim link, it pre-builds signed TC packets with `build_tc_packet` and passes them as
   `inbound_packets`.
5. It creates bus subscriptions before the first step.
6. Each `step` passes `dt = now - clock` into `ValidationHarness.step`. `step_once`
   advances the clock in inner chunks.
7. `collect` drains subscriptions, reads gimbal position with a noise tolerance, and polls
   emulator downlink on real link.
8. `shutdown` closes the emulator and station link.

## Errors and faults

`InProcessBackend` raises `RuntimeError` when `step` or `collect` run before `build`.
`SocketBackend` always raises `NotImplementedError`.

## Messages

Backends subscribe passively to `InferenceResultMsg`, `GimbalCommandMsg`, `ModeChangeMsg`, and
`CommandAckMsg`. They do not publish.

## Configuration

Profile TOML selects `EnvironmentConfig` axes. Default SIL uplink key is
`b"sil-test-key-0000000000000000000"`.

## Constraints

- This module is a stub backend holder for `SocketBackend`. PIL/HIL transport is not run.
- Sim-link `inject_command` is intentionally a no-op. All pre-baked packets drain on step 1.
- Gimbal-moved uses 0.1 deg tolerance to ignore encoder noise.
- Imports are limited to `flight.libs` and `sim`.

## Related documents

- [`gse`](gse.md)
- [`gse.orchestrator`](orchestrator.md)
- [`gse.scenario`](scenario.md)
- [`gse.station`](station.md)
- [`sim.sil.stepping`](sim/sil/stepping.md)
