# `sim` Subsystem Context

Non-obvious context for the SIL (software-in-the-loop) sim package. Documents
cross-cutting decisions not derivable from the individual files or their docstrings.

---

## SIL runs the REAL flight apps -- no parallel wiring

`build_sil_system` constructs sim drivers, bundles them as `flight.core.composition.Drivers`,
and wires them through the **same** `build_apps(...)` the flight entry (`flight/core/main.py`)
uses. There is deliberately no SIL-specific app graph. The only difference between flight and
SIL is the `Drivers` bundle (sim vs. real HAL) and the clock (`ManualClock` vs. `RealClock`).
Consequence: any wiring change in `composition.py` is exercised by SIL for free; do not
re-implement app construction here.

## Byte-level station link in SIL (ADR 0009)

- `build_sil_system` accepts `inbound_packets: list[bytes]` (raw CCSDS-framed byte strings)
  and `uplink_key: bytes` (defaults to the SIL test key
  `b"sil-test-key-0000000000000000000"`). These are passed to `SimStationLink` and `build_apps`
  respectively.
- Command-path SIL tests build signed packets with `build_tc_packet`, pass them as
  `inbound_packets`, and subscribe to `CommandMsg` / `CommandAckMsg` on the bus to assert
  acceptance or rejection. The SIL test key must match the key used in `build_tc_packet`.
  **Canonical import:** `from flight.libs.commands import build_tc_packet` -- the symbol was
  relocated there so the command codec lives beside the command dictionary it serializes. The old
  `flight.iss_iface.ingress` path is a back-compat re-export only; new code (SIL tests and the GSE
  `StationEmulator` alike) must import from `flight.libs.commands`.
- The in-process station emulator seam is `SimStationLink` (from `flight.hal.drivers_sim`).
  A full ground-support emulator (`packages/gse`) is deferred future work -- do not assume
  one exists.

## Config-matrix axes and the `step_once` seam (validation effort)

The validation venues are driven by `flight.libs.config.config.EnvironmentConfig` -- five
per-axis `AxisMode` knobs (`sensor`/`gimbal`/`compute`/`link`/`clock`, each `"sim"` or `"real"`)
plus `host`. `profiles/*.toml` are config **overrides** applied via
`load_config("config/default.toml", "profiles/NAME.toml")`: `sil` (all sim), `sil-link-real`
(link real, rest sim) are the **running** venues; `pil`/`hil` are **DEFINED-NOT-RUN** (see
`docs/validation/`). `flight.core.select_drivers.select_drivers(config, clock, sim_inputs)` maps
each axis to a sim or real driver; the clock axis is resolved by the composition root *before*
calling it.

The single-step body of `SilHarness.step` is extracted verbatim into
`sim.sil.stepping.step_once(...)` -- a driver-agnostic, Protocol-typed function. `SilHarness.step`
delegates to it, and the GSE `InProcessBackend` reuses the **same** `step_once` so the in-process
validation harness and the SIL harness share one stepping implementation. Do not fork a second
stepper.

The `lock` axis (LaunchLock) is intentionally absent from `EnvironmentConfig`: no device exists,
so it is a permanent VCRM gap, not a config knob.

## The harness is a single-threaded stepper -- it replaces the scheduler, not the apps

Flight runs the apps under a thread `Scheduler` whose per-subsystem `run()` loops both do the
work *and* emit heartbeats. `SilHarness` has no threads and no scheduler: each `step(now)`
directly drives the app methods (`process_frame`, `sample`, `tick`) in a fixed order, advancing
`now` explicitly for full determinism. Because the `run()` loops never execute, **nothing emits
heartbeats** -- so the harness manually publishes one liveness `HeartbeatMsg` per
`MONITORED_SUBSYSTEMS` entry every step (`sequence=0`, hardcoded). Drop that loop and the FDIR
watchdog trips `PROCESS_DIED` within three steps. The harness must stay synchronized with
`MONITORED_SUBSYSTEMS`; a new monitored subsystem needs a matching heartbeat here.

## Scene renders radiometrically-plausible mosaic frames (1024 as of 2026-06-11)

`scene/plume.py:build_frames(num_frames, seed)` renders **raw 1024x1024 uint16 mosaic frames**
(512 band planes): background + Gaussian plume in band-plane space, interleaved back into the 2x2
CFA mosaic via `interleave_bands`, quantized to 12-bit. The NIR plane is brighter inside the plume
region (smoke reflects strongly in NIR), matching the Sentinel-2-derived training domain. The
scene is deterministic for a given `seed`. The plume sits **off-center** at band-plane (340, 340)
-- ~119 px from the (256, 256) boresight, above the minimum deadband and below the maximum -- so
TRACKING issues rate commands that point the gimbal toward it (the command-direction proof). In
decimated search mode (scale 0.5) it appears at tensor ~(170, 170), inside the scripted mask
`[145:195, 145:195]`.

The SIL closed-loop tests now run real signal through the full ingest path:
`calibrate_mosaic -> separate_bands -> normalize_dn -> select_bands -> compute_quality_flags ->
ScriptedDetector`. `ScriptedDetector` still detects from its fixed probability mask (not tensor
content), so the closed-loop test result is unchanged; the value is that domain drift in the
ingest path becomes visible in the SIL rather than only at HIL.

## The harness advances the clock so SimGimbal dynamics integrate (ADR 0008)

`SilHarness.run_steps(count, dt)` advances the shared `ManualClock` by `dt` each step. This is
load-bearing: `SimGimbal` integrates its first-order dynamics *lazily* on elapsed clock time, so
without the per-step advance the gimbal never moves and the closed-loop assertions are vacuous.
`step()` drains `payload.poll_mode_changes()` and passes the latest `read_position()` and the
SAFE flags into `process_frame`. `payload_gimbal_state()` is a test/inspection accessor for the
arbiter's current state. The closed-loop tests assert the *mechanism and direction* (thermal SAFE
-> stow switch closes; ground `ModeChangeMsg(IDLE)` un-latches SAFE; TRACKING drives +az/-el
toward the off-center plume), **not** photometric convergence -- see the twin note below.

**Identity calibration in SIL:** `build_sil_system` builds a `MosaicCalibration` with zero
dark / unit flat / no bad pixels via `calibration_io.build_identity_calibration`, then passes it
into `build_apps` as `calib=`. `SensorConfig.calibration_dir = ""` in the default TOML selects
this path; flight sets a real directory. Do not supply a real `calibration_dir` in SIL/tests
unless you also provision the artifact files.

## SIL closed-loop tests are in the default CI gate

`tests/test_sil_closed_loop.py` is **not** marked `e2e`, so it runs under the standard
`pytest -m "not e2e"` CI job -- the deterministic, thread-free design makes the full closed loop
cheap enough to gate every commit.

## `sim/twin` is a deferred scaffold

`sim/twin/` is empty by design. `SimGimbal` integrates first-order az/el dynamics (ADR 0008),
which is enough to prove command direction and the SAFE-stow mechanism, but the **scene is
static** -- the rendered plume does not move in response to gimbal motion. So the closed-loop
tests assert command direction/mechanism, not that tracking *converges* on the plume. A real
dynamics twin that closes the scene-feedback loop is future work; do not assume one exists or
write tests that depend on photometric convergence.
