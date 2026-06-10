# PACT Flight Software -- Required Final State (Design Spec)

- Date: 2026-06-09
- Status: Approved (brainstorm validated section-by-section with the user)
- Branch: fsw-restructure
- Inputs: `docs/superpowers/baseline/2026-06-06-pact-flight-parity-baseline.md` (the "current"
  side of the delta) and `docs/superpowers/specs/2026-05-30-pact-iss-payload-fsw-structure-design.md`
  (the structure spec, amended by Section 10 of this document).
- Scope: the required final state of the flight software, its external interfaces, the data
  system, the requirements/V&V baseline, and the SIL -> PIL -> HIL validation ladder. This spec
  defines WHAT the final state is; implementation plans (per interface/phase) define the work.

---

## 1. Framing and posture

PACT remains an ISS-attached payload for autonomous plume detection, segmentation, and tracking,
on a single Linux flight computer, in Python, with the subsystem-app + typed-bus architecture.
Nothing in this spec changes the architectural invariants (app isolation, pure cores, HAL
Protocols, composition-root ownership, preprocessing co-location, Result[T, E]).

**Process rigor target: NASA-payload-grade, pragmatic** -- modeled on NPR 7150.2 Class C/D
payload software:

- A consolidated requirements baseline with IDs, rationale, and verification traceability
  (Section 8). No DO-178-style certification artifacts.
- Hazard-driven safety inhibits as the one non-relaxed area. The three hazardous functions are
  **gimbal motion** (stored mechanical energy), **launch-lock release**, and **thermal limits
  affecting the host**. Software obligations: honor inhibits, never actuate a hazardous function
  without authorization, verify inhibits by test at every rung of the validation ladder.
- Reliability posture unchanged: fail-safe, ground-recoverable, graceful degradation.

**Hardware reality:** a full bench rig will exist but does not yet. This spec nominates concrete
reference hardware (Section 2) so drivers, calibration, and the HIL bench are designed against
real datasheets; everything stays swappable behind the HAL.

---

## 2. Reference hardware

| Element | Reference | Interface |
|---|---|---|
| Camera | FLIR Blackfly S monochrome (Sony IMX-class, 12-bit) behind a custom 2x2 mosaic filter | GigE, PySpin SDK |
| Mosaic filter | 2x2 tile, passbands approximating Sentinel-2 B2/B3/B4/B8 (490/560/665/842 nm) | -- |
| Gimbal | FLIR PTU-5-class two-axis pan/tilt with encoders | Serial (RS-232 / Ethernet ASCII) |
| Launch lock | Motorized locking pin with engaged/released microswitches | GPIO/serial |
| Flight computer | NVIDIA Jetson Orin NX-class, Linux | -- |
| Station link | Ethernet at a Bartolomeo-class external payload site | CCSDS Space Packets over UDP/TCP |
| HK sensors | Board temperatures / currents | I2C / sysfs |

The mosaic passbands deliberately approximate the Sentinel-2 bands so the external training
dataset (Section 4) remains a valid training domain.

---

## 3. Sensor ingest chain

**HAL contract change (the deepest fix).** `ImagingSensor.acquire_frame()` returns a raw
`(H, W)` uint16 **mosaic plane** plus timestamp/exposure/gain metadata. Drivers acquire only --
no image processing inside any driver. The control plane (`set_exposure`, `set_gain`, ROI) stays
on the Protocol. `RealSensor` becomes a full PySpin acquisition + node-map control driver with
lazy SDK import; `SimSensor` replays raw mosaic frames rendered by `sim.scene`.

**Frames never touch the bus.** The payload app calls its injected sensor driver directly
(co-location invariant preserved); `RawFrameMsg`-style band stacks on the bus are removed.

**Preprocess pipeline** (pure functions in `flight/payload/preprocess/`, in order):

```
bad-pixel mask -> per-pixel dark/flat correction (from checksummed calibration files)
  -> demosaic: 2x2 CFA separation into 4 registered band planes
  -> normalization to a [0, 1] reflectance-like domain
  -> quality gates (saturation, motion smear from exposure x slew rate, illumination)
  -> ROI crop (re-enabled)
```

- **Demosaic lives in preprocess**, not the driver: testable pure code, calibration applied on
  the raw plane where it physically belongs, and the SIL exercises the full ingest path.
- **Band vocabulary**: `B2/B3/B4/B8` renamed to `BLUE/GREEN/RED/NIR`, with the Sentinel-2
  correspondence documented at the definition site.
- **Calibration artifacts** (dark, flat, bad-pixel) live in `data/calibration/`, loaded and
  hash-verified at startup. Identity calibration becomes SIL-only.
- Quality heuristics become physically grounded (smear from commanded slew rate x exposure,
  illumination from scene statistics in the normalized domain).

---

## 4. Model lifecycle: external training, artifact intake here

Training, dataset handling, and sensor-model augmentation live in a **separate model repository**.
No torch in this repository, ever; the legacy `src/pact/model` torch stack is removed with legacy
retirement, not migrated. Training data: the NeurIPS 2020 CCAI workshop dataset
(*Characterization of Industrial Smoke Plumes from Remote Sensing Data*, Sentinel-2-derived).

**The model output contract is binary segmentation + blobs**: a `(1, 1, H, W)` sigmoid plume
mask, thresholded, then `extract_blobs` derives centroids/boxes for tracking. The mask itself is
a science product (stored + downlinkable as compressed thumbnails).

**Artifact contract** (defined in this repo, honored by the model repo): a frozen `.onnx` plus a
sidecar manifest -- semantic version, model-repo git SHA, training-dataset hash, input contract
`(1, 4, H, W)` float32 in the normalized domain produced by `flight/payload/preprocess`, output
contract `(1, 1, H, W)` sigmoid, SHA-256 of the artifact.

**Acceptance harness** (`tools/`): validates manifest + hash + I/O contract, runs the artifact
(onnxruntime only) against a golden SIL scene set with an eval gate (minimum mask IoU), and
checks the per-frame latency budget. Passing acceptance is what admits an artifact into
`data/models/`. INT8 quantization happens in the model repo; its eval gate runs here, tuned by
PIL latency measurements (Section 9).

**Domain alignment**: the model repo must replicate flight preprocessing at train time, so
`pact-sim`'s camera model and the flight demosaic/preprocess functions are its dependency
(installable from this repo via git). One implementation, two consumers, no silent domain drift.

**Flight side** (`OnnxDetector`): loads `.onnx` only (the `.pt` default is removed); verifies
hash + I/O contract at load (`MODEL_CORRUPT` on mismatch); telemeters version + hash; enforces a
per-frame latency budget (`INFERENCE_TIMEOUT` fault, frame dropped, loop continues). CI keeps a
tiny checked-in fixture `.onnx` so the detector is behaviorally tested.

---

## 5. Pointing, gimbal, mechanical, SAFE, FDIR

### Gimbal HAL (closed-loop)

`GimbalActuator` grows to the PTU-class command set: `goto_angle(az, el)`,
`set_rate(az_rate, el_rate)`, `home()`, `stow()`, `read_position()` returning timestamped
encoder angles, and stow-switch state. Travel/slew/accel limits are enforced in the driver AND
checked in the arbiter (defense in depth). `SimGimbal` gains first-order dynamics (rate limits,
lag, encoder noise) so the closed loop is honest in SIL. `RealGimbal` becomes a real serial
driver for the reference PTU.

### Pointing math (fixes the silently-wrong path)

- Error is **boresight-relative**: `centroid - frame_center`, converted pixel -> line-of-sight
  through camera intrinsics (focal length / IFOV from the reference optics), then LoS -> gimbal
  frame. The hard-coded `PIXEL_TO_DEG` scalar is removed.
- The LQR tracks a setpoint (target at boresight) and outputs **rate** commands during TRACK;
  ACQUIRING/SCAN/stow use absolute `goto_angle`.
- Safety gates are wired into the live `PayloadController` path: `check_deadband`, slew-rate
  limiting, release hysteresis, strike counting. Runaway detection becomes physical:
  commanded-vs-encoder divergence over a window, not pixel inference.

### Mechanical = launch lock

The `mechanical` app owns a `LaunchLock` HAL device (motorized pin, engaged/released
microswitches): `release()`, `engage()`, `read_state()`. Lock release is a hazardous command --
two-step ARM then EXECUTE, inhibit-gated, ground-only. The interlock runs both ways: the
arbiter/gimbal driver refuse motion while the lock reads ENGAGED, and lock state is on the bus +
telemetered. SAFE does not re-engage the lock; re-engagement is a ground-commanded
end-of-mission operation. There is no aperture cover.

### SAFE mode (single tier, latched, ground exit)

On `ModeChangeMsg(SAFE)`: the arbiter enters its SAFE state and commands a stow (the one
mechanical safing action); payload halts acquisition/inference; thermal falls back to survival
setpoints; telemetry and fault annunciation continue. SAFE latches (debounced, no auto-exit);
only a ground `EXIT_SAFE` command, gated on the triggering fault being cleared, restores
NOMINAL.

### FDIR hardening

The fault app emits its own heartbeat (monitored by the scheduler); whole-process death is the
external supervisor's job (Section 7). Faults persist to a storage-backed ledger that survives
reboot and are downlinked as events (ground annunciation); the active-fault set + system mode
are in telemetry.

---

## 6. ISS interface, command path, and the data system

### Link transport

`RealStationLink` becomes a real Ethernet transport for a Bartolomeo-class site: CCSDS Space
Packets -- commands inbound over TCP, telemetry/products outbound over UDP -- with CRC, per-APID
sequence counts, and AOS/LOS state driven by the station. Link state is published on the bus;
the downlink manager drains only during AOS.

### Command ingress (`iss_iface`)

Decode -> CRC check -> sequence dedup -> source authentication (HMAC over the command packet) ->
validation against a typed **command dictionary** (per-target command IDs + param schemas,
defined in `flight.libs`). Only validated commands become `CommandMsg`; every inbound command
produces an ACK or NACK downlink event, always.

### Command routing (`core`)

A command-router service dispatches by target and tracks completion: accept/reject at dispatch,
execution result acked by the target app, unknown target/command -> loud NACK + fault event (no
silent drops). Hazardous commands (lock release, manual gimbal slews, `EXIT_SAFE`) carry the
ARM/EXECUTE two-step and are re-checked **at the point of actuation** against fault-owned
inhibit state -- the layered authority model: iss_iface validates, core routes, actuating apps
enforce inhibits.

### Storage (core service)

File-backed, checksummed, quota'd with retention policy. Two faces:

- a bus consumer persisting telemetry, events, and the fault ledger;
- a direct-call `StorageWriter` Protocol injected into the payload app, so masks/thumbnails
  bypass the bus (large-artifact invariant).

### Downlink manager (core service)

A priority queue -- fault events > command acks > HK telemetry > science products -- drained
within configured comms budgets and AOS windows. Bus messages carry only compact references
(storage entry IDs); `iss_iface` receives an injected `StorageReader` Protocol to fetch product
bytes at transmission time.

### Model upload

Chunked uplink commands -> `iss_iface` reassembles -> staged into storage -> manifest SHA-256 +
load-validation acceptance -> ground `ACTIVATE` swaps the active artifact, with automatic
rollback if the new model fails to load or fails its first-frame sanity check ->
`ModelDeployState` telemetered throughout.

---

## 7. Platform robustness

### Process model (formalized; amends the 2026-05-30 spec)

One flight process, thread-per-app, `queue.Queue` bus -- what is actually built and documented
in ADR-0003. The old spec's "multiprocessing.Queue, permanently" clause is superseded (Section
10). The missing discipline is added instead:

- **Bounded queues** with per-message-type depth and explicit overflow policy: drop-oldest +
  drop counter for telemetry; never-drop for commands/faults, where overflow is itself a fault
  event.
- Envelopes gain a `schema_version` field.

### Startup / shutdown / supervision

- `main()` handles SIGTERM with ordered teardown: quiesce payload -> drain downlink -> flush
  storage -> join threads.
- Startup **health gate**: all monitored apps must heartbeat within a configured window before
  the system declares NOMINAL; otherwise it enters SAFE and annunciates.
- The scheduler supervises app threads: a crashed thread restarts up to a configured restart
  limit, then FDIR latches SAFE.
- Whole-process death is caught by an external supervisor (systemd unit with watchdog notify) --
  the canonical two-layer watchdog.

### Config integrity

`config_loader._validate()` becomes real: range checks, cross-field checks (e.g. band indices
vs mosaic layout), unknown-key rejection so typos fail loudly at startup (the one place raising
is correct). `main()` honors the `flight.toml` override path. The defaults-vs-TOML drift test is
retained.

---

## 8. Requirements + V&V baseline

- `docs/requirements/`: one document per subsystem plus a system-level document. Each
  requirement carries an ID, rationale, and **verification method + rung** (unit / SIL / PIL /
  HIL).
- Traceability, lightweight but enforced: module docstrings cite the REQ IDs they satisfy
  (existing convention); tests mark the REQ IDs they verify; a CI script asserts every
  requirement is cited by at least one module and one test.
- A short **hazard analysis** document covers the three hazardous functions (gimbal motion,
  launch-lock release, thermal limits affecting the host), tracing each hazard to its software
  inhibits and their verifying tests.

---

## 9. Validation ladder: SIL -> PIL -> HIL, one shared harness

### `packages/gse` (new package: ground support equipment)

One harness drives all three rungs, so a scenario written once climbs the ladder unchanged:

- **Station emulator**: the station side of the exact protocol `RealStationLink` speaks --
  CCSDS command uplink, telemetry/product capture, AOS/LOS window scheduling. In-process for
  SIL; on a bench PC over real Ethernet for PIL/HIL.
- **Scenario format**: declarative files -- config overrides, scene/plume script, command
  timeline, expected-outcome assertions ("TRACKING within N frames", "SAFE latched", "product
  downlinked with valid CRC").
- **Orchestrator + analysis**: runs scenarios, collects telemetry/logs, scores assertions, and
  produces the V&V evidence the requirements matrix points at.

Layering: `gse` imports `flight.libs` (message/packet definitions) and `sim`; `flight` never
imports `gse`.

### SIL (every CI run)

The existing harness, deepened to the new contracts: `sim.scene` renders physically-grounded
**raw mosaic** frames (plume radiance, band responses, PSF, sensor noise); `sim.twin` gains
gimbal dynamics, plume/scene kinematics, and a link-impairment model (latency, drops, AOS/LOS).
Deterministic with `ManualClock` for CI; also runs wall-clock for interactive use.
Retires: logic, contracts, control-loop correctness.

### PIL (on the Jetson, before hardware exists)

The unmodified flight image on the Jetson Orin NX-class target -- sim drivers for
sensor/gimbal/lock, but the **real** station link over real Ethernet to GSE on a bench PC.
Retires: aarch64 issues, true onnxruntime latency (feeds the quantization eval gate), CPU/thermal
load, real network-stack behavior.

### HIL (the future bench)

Real camera imaging a controllable scene (display/projector target), real PTU gimbal + launch
lock on the bench, GSE as the station, all `drivers_real` selected by config.
Retires: driver correctness, optics + calibration reality, closed-loop pointing on real
dynamics, end-to-end command/safety/downlink paths.

Each requirement's verification rung (Section 8) maps onto this ladder.

---

## 10. Architecture packaging and amendments to the 2026-05-30 spec

**Packaging (Approach A, chosen over cFS-style peer apps and lean-onboard alternatives):**
command router, storage, and downlink manager are **services hosted by `flight.core`** (per the
original structure spec); each heartbeats and is watchdog-monitored. `mechanical` is a real peer
app (launch lock). Model upload spans `iss_iface` (reassembly) and `core` (stage / activate /
rollback). `packages/gse` is a new workspace package.

**Amendments to the 2026-05-30 structure spec:**

1. Spec Section 5 "transport is `multiprocessing.Queue` (pickle), permanently" is superseded:
   the transport is in-process `queue.Queue`, thread-per-app, one flight process (consistent
   with ADR-0003 and the implementation), now with bounded queues + overflow policy.
2. `mechanical` is re-chartered from "covers / deployables / latches" to the launch lock (there
   is no aperture cover).
3. `tools/` is re-chartered from "training / experiments" to artifact acceptance + SIL
   experiment runners + analysis (training lives in the external model repo; no torch here).
4. The repo layout gains `packages/gse/`.

**Legacy retirement** is part of the final state: `src/pact/` removed, CI gates widened to the
whole tree. (Execution timing remains user-gated.)

### ADRs to write

1. Sensor ingest contract: raw-mosaic HAL + demosaic in preprocess + band vocabulary.
2. Model lifecycle: external training repo, artifact contract, acceptance gate, binary
   segmentation + blobs.
3. ISS interface: Ethernet + CCSDS reference, layered command authority
   (validate / route / inhibit-at-actuation), ack/NACK contract.
4. Gimbal: absolute + rate closed-loop command model; boresight-relative pointing; launch-lock
   interlock.
5. SAFE semantics: single latched tier, ground-commanded exit.
6. Data system: core-hosted storage + prioritized downlink + model upload (Approach A).
7. Process model: threads-in-one-process formalized; bounded queues; two-layer watchdog
   (supersedes the old spec's process-per-app clause).
8. Validation ladder: SIL -> PIL -> HIL with a shared GSE harness.

---

## 11. Decision log (2026-06-09 brainstorm)

| Question | Decision |
|---|---|
| Hardware reality | Full bench rig will exist, none today; plan full HIL integration |
| Hardware selection | Nominate reference hardware in this design (Section 2) |
| Process rigor | NASA-payload-grade pragmatic (NPR 7150.2 Class C/D shape) |
| Demosaic location | Preprocess owns it; HAL returns raw mosaic |
| Model output | Binary segmentation + blobs; mask is a science product |
| Training | External model repo; NeurIPS 2020 CCAI smoke-plumes dataset; artifact intake here |
| ISS link | Ethernet + CCSDS Space Packets (Bartolomeo-class reference) |
| Command authority | Layered: iss_iface validates, core routes, inhibit checks at actuation |
| SAFE | Single latched tier; stow + quiesce + survival setpoints; ground exit gated on fault clear |
| Gimbal model | Absolute + rate, closed-loop encoder feedback, home/stow, limits in driver + arbiter |
| Mechanical | Launch lock (no aperture cover); release is ARM/EXECUTE hazardous command |
| Data system | Full chain: storage + prioritized downlink + model upload |
| Process model | Threads-in-one-process formalized; old spec amended |
| Validation | SIL -> PIL -> HIL ladder with shared GSE harness (new packages/gse) |
| Packaging | Approach A: core-hosted services + GSE package |
