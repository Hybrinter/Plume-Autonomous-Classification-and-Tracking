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

## The harness is a single-threaded stepper -- it replaces the scheduler, not the apps

Flight runs the apps under a thread `Scheduler` whose per-subsystem `run()` loops both do the
work *and* emit heartbeats. `SilHarness` has no threads and no scheduler: each `step(now)`
directly drives the app methods (`process_frame`, `sample`, `tick`) in a fixed order, advancing
`now` explicitly for full determinism. Because the `run()` loops never execute, **nothing emits
heartbeats** -- so the harness manually publishes one liveness `HeartbeatMsg` per
`MONITORED_SUBSYSTEMS` entry every step (`sequence=0`, hardcoded). Drop that loop and the FDIR
watchdog trips `PROCESS_DIED` within three steps. The harness must stay synchronized with
`MONITORED_SUBSYSTEMS`; a new monitored subsystem needs a matching heartbeat here.

## Scene renders radiometrically-plausible mosaic frames (as of 2026-06-09)

`scene/plume.py:build_frames(num_frames, seed)` renders **raw 512x512 uint16 mosaic frames**:
background + Gaussian plume in band-plane space, interleaved back into the 2x2 CFA mosaic via
`interleave_bands`, quantized to 12-bit. The NIR plane is brighter inside the plume region
(smoke reflects strongly in NIR), matching the Sentinel-2-derived training domain. The scene
is deterministic for a given `seed`.

The SIL closed-loop tests now run real signal through the full ingest path:
`calibrate_mosaic -> separate_bands -> normalize_dn -> select_bands -> compute_quality_flags ->
ScriptedDetector`. `ScriptedDetector` still detects from its fixed probability mask (not tensor
content), so the closed-loop test result is unchanged; the value is that domain drift in the
ingest path becomes visible in the SIL rather than only at HIL.

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

`sim/twin/` is empty by design. The SIL gimbal (`flight.hal.drivers_sim.SimGimbal`) just
integrates az/el deltas in software, which is sufficient for current closed-loop tests. A real
dynamics twin is future work; do not assume one exists.
