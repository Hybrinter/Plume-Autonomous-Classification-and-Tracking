# PACT Flight Software -- Parity Baseline (current state)

- Date: 2026-06-06
- Branch: fsw-restructure
- Status: Baseline assessment (the "current" side of a current-vs-final delta)
- Method: 14 parallel Opus subsystem surveys + 4 Opus external-interface syntheses over
  `packages/flight`, `packages/sim`, `packages/tools`, and the design docs; highest-impact
  findings spot-verified against source.
- Purpose: establish what genuinely exists today so a follow-on spec + ADR can define the
  delta to a flight-real implementation that interfaces with the ISS, the gimbal, the
  detection/segmentation models, and the monochrome + 2x2 Bayer/mosaic sensor.

This document records the BASELINE only. It deliberately does not propose solutions; the
required-final-state and the rationale belong in the forthcoming spec and ADR(s).

---

## 1. Executive summary

The subsystem-app **skeleton is real and good**: the typed bus, the pure-core/thin-shell
discipline, the Protocol HAL with composition-root injection, the config loader + default-drift
guard, the FDIR watchdog/policy cores, the tracking math (EMA, constant-velocity Kalman, IoU
matcher), the LQR (solved from DARE), and the deterministic closed-loop SIL all exist, are
unit-tested, and are CI-green. The architecture is sound.

The **flight capability is largely absent**. Three categories of gap dominate:

1. **All four real external drivers are stubs** -- `RealSensor` (returns `CAMERA_STALL`),
   `RealGimbal` (drops commands, reads origin), `RealStationLink` (inert), `RealScalarSensor`
   (returns `0.0`). The flight entry `main()` wires them but has never run end-to-end on
   hardware, so the production system is **wired-but-inert**.

2. **Whole subsystems the design calls for do not exist**: no on-board **storage**, no
   **downlink product** pipeline (nothing ever produces a `DownlinkItemMsg`), no **command
   router/handlers** (a `target='payload'` command is silently dropped), no **SAFE actuation**
   (nothing subscribes to `ModeChangeMsg`), no **model-upload** subsystem (`UploadChunkMsg` is
   orphaned), no **ML training/export** in `tools/` (it is a 0-byte package; no `.onnx`
   artifact exists anywhere), no **thermal/electrical control** (monitor-only), and an empty
   **`mechanical`** scaffold.

3. **Several wired paths are silently wrong** -- more dangerous than the known stubs because
   tests pass and SIL is green:
   - **Pointing uses absolute pixel position as error** (`arbiter.py:216`): a plume centered
     at (128,128) of a 256-px frame commands a ~5 deg slew instead of ~0. The LQR likewise
     drives the estimate to zero with no target setpoint (`control.py:127`).
   - **Safety gates exist, are tested, and are never called** in the live path:
     `check_deadband` (min-displacement + `GIMBAL_RUNAWAY`) and slew-rate limiting are not
     wired; `max_slew_rate_deg_per_s`, `release_persistence_frames`, and
     `max_deadband_strike_count` are loaded from config but unused.
   - **SAFE is a no-op**: `ModeChangeMsg(SAFE)` is published but no production app consumes it,
     so a watchdog/thermal/runaway fault never actually stows the gimbal or quiesces the
     payload. `exit_safe_mode` is never invoked -- there is no enter OR leave SAFE in flight.
   - **`RealScalarSensor` 0.0** means thermal reads 0 C and can never trip `THERMAL_OVER_LIMIT`
     on real hardware (a silent-failure inversion of the fail-loud `RealSensor`).
   - **`model_path` defaults to a `.pt`** fed to `onnxruntime`; `main()` constructs
     `OnnxDetector` without a `model_version` (telemeters `"unknown"`) and never passes the
     `flight.toml` override path; `config_loader._validate()` is a no-op so out-of-range or
     typo'd config loads silently.

The single deepest cross-cutting issue is the **sensor-domain mismatch** (Section 4.4): every
layer assumes Sentinel-2-style named bands (B2/B3/B4/B8) and already-separated `(4,H,W)`
stacks, but the physical device is a **monochrome camera behind a 2x2 Bayer/mosaic filter**.
There is **no demosaicing anywhere** (repo-wide: zero hits for demosaic/debayer/bayer/mosaic/
CFA). This contradicts messages, config, preprocessing, the SIL scene, and the (legacy) training
domain at once -- it is a contract change, not a localized fix.

---

## 2. What genuinely works today (the foundation to preserve)

- **Bus + contract** (`libs`): typed pub/sub routed by exact message type; 12 frozen message
  dataclasses + `BlobMeta`; 7 enums incl. `FaultCode`; `Result[T,E]`; `Clock`
  (monotonic/wall) with `Real`/`Manual` impls; config loader with happy/error paths and a
  default-vs-TOML drift test. All unit-tested.
- **HAL seam**: every device is a `@runtime_checkable` Protocol returning `Result[..,FaultCode]`
  with lazy SDK imports; `build_apps` is genuinely driver-agnostic and reused verbatim by SIL.
- **Payload pure cores**: `select_bands`, radiometric `(raw-dark)/flat` with finite-check, five
  quality heuristics, ROI crop/backproject; EMA filter; constant-velocity Kalman
  (predict/update, singular-S guard); IoU `compute_iou` + greedy `match_blobs` with persistent
  IDs; `GimbalArbiter` FSM (IDLE/ACQUIRING/TRACKING/SCAN/SAFE) with rate limiting; LQR gains
  solved from `solve_discrete_are`. ~44 payload-control + 26 tracking tests pass.
- **Detector backend**: `DetectorBackend` Protocol; `ScriptedDetector` (deterministic) and a
  genuinely-wired `OnnxDetector` (lazy onnxruntime, sigmoid, `INFERENCE_NAN` on non-finite,
  `perf_counter` timing); shared `extract_blobs` (scipy connected components).
- **FDIR cores**: pure heartbeat watchdog (miss-count -> `WATCHDOG_EXPIRE`) and static
  `SAFE_TRIGGERING_FAULTS` policy; thin `FaultApp` shell. Tested end-to-end on the bus.
- **Housekeeping templates**: `ThermalApp`/`ElectricalApp` (sample -> telemetry -> over-limit
  self-report -> heartbeat -> command-ack no-op); over-limit genuinely routes to SAFE in tests.
- **Composition + SIL**: thread `Scheduler`; `SilHarness` deterministic single-threaded stepper
  reusing the real `build_apps`; two closed-loop tests (nominal plume->gimbal-command; thermal
  over-limit->SAFE) run in the default CI gate.

---

## 3. Per-subsystem maturity

| Subsystem | Skeleton | Flight-real behavior | Headline gap |
|-----------|:--------:|:--------------------:|--------------|
| libs (bus/messages/types/config/time) | strong | partial | orphaned envelopes (Storage/Upload/Downlink), no schema version, no ack/seq, unbounded queues |
| hal interfaces | strong | n/a | contracts thin (gimbal delta-only; sensor presupposes separated bands) |
| hal drivers_real | present | **none** | sensor/gimbal/station/scalar all stubs |
| hal drivers_sim | strong | sim-only | no dynamics/limits/noise/link-impairment |
| payload/preprocess | strong | **sim-only** | no demosaic; identity calibration; no normalization; crop disabled |
| payload/model | strong | **stub** | no `.onnx` artifact; detection-only (no segmentation); no provenance/timeout |
| payload/tracking | strong | simplified | single-target; pixel-space only; fixed dt; `PIXEL_TO_DEG` scalar |
| payload/control | strong | **wrong/partial** | absolute-centroid error; gates not wired; SAFE no stow; no rate limit |
| fault | strong | partial | SAFE unactioned; no persistence/ground annunciation; FDIR self-unmonitored |
| iss_iface | strong | **stub** | inert real link; no auth/inhibit/ack/routing; no downlink producer; no upload |
| thermal/electrical | strong | monitor-only | 0.0 real sensor; no heater/EPS control |
| mechanical | **absent** | absent | empty package (covers/deployables/latches + inhibits) |
| core | strong | partial | no storage/downlink-aggregator app; no validation; main untested on HW; no SIGTERM/supervision |
| sim | strong | low-fidelity | zeroed frames + fixed mask; empty twin; no plant/orbit/link physics |
| tools | **absent** | absent | 0-byte package; no ONNX export anywhere; quantize is a no-op stub; legacy unmigrated |

---

## 4. The four external interfaces -- current vs flight-real

### 4.1 ISS station link
**Current.** Pure in-process transport bridge: `IssIfaceApp.pump_uplink` republishes inbound
`CommandMsg` verbatim; `pump_downlink` drains `DownlinkItemMsg` to the link. `RealStationLink`
is inert (`receive_command -> Ok(None)`, `send_downlink` drops). The wire protocol is
explicitly deferred (ADR-0006 scopes out RF/CCSDS; spec Sec 15 lists the avionics interface as
open). No auth, no authorization, **no safety-inhibit gate**, no seq ordering/dedup/replay, no
CRC, no ack/NACK. Only thermal/electrical subscribe `CommandMsg` (ack-only no-op); payload/
core/fault do not -- a `target='payload'` command is silently dropped. **No flight app ever
produces a `DownlinkItemMsg`** (test-only), `CommsConfig` caps/windows are unenforced, and
model upload (`UploadChunkMsg`/`ModelDeployState`) is orphaned. Faults/SAFE are never downlinked.
**Final-real needs.** Real avionics transport in `RealStationLink`; command ingress validation
+ auth + **hazardous-command inhibit**; command router + acting handlers + ack/NACK; downlink
product pipeline (priority queue, CRC, packaging) + storage feeding it + comms-budget/contact
windows; model-upload subsystem; fault/SAFE ground annunciation + ground-commanded SAFE-exit;
decoupled poll vs heartbeat cadence; link-impairment sim model.

### 4.2 Gimbal pointing
**Current.** `tracking -> arbiter FSM -> LQR -> safety -> GimbalActuator`. `RealGimbal` drops
commands and reads origin; `SimGimbal` is a perfect instantaneous delta-integrator (no
dynamics/limits/noise). The loop is **open-loop** (`read_position` is never consumed; runaway is
inferred from image pixels, not encoder feedback). Command model is **delta-only** (no absolute/
rate, no limits, no homing/stow). Pointing error uses **absolute** pixel centroid; deadband/
runaway/slew-rate gates are unwired; SAFE merely stops commanding (no stow); release hysteresis
and strike-count unimplemented; `PIXEL_TO_DEG=0.04` is a single hard-coded scalar (no
intrinsics/FOV/distortion).
**Final-real needs.** Real serial/CAN driver with timestamped encoder feedback + fault
reporting; richer command model (absolute/rate, travel/rate/accel limits, homing/stow);
boresight-relative error + real pixel->LoS->gimbal-frame transform + target setpoint;
closed-loop divergence/runaway on encoder feedback; enforced slew limits + deadband + SAFE stow
+ inhibits; a gimbal+orbit dynamics twin.

### 4.3 ML detection + segmentation
**Current.** Preprocess -> `ProcessedFrameMsg` (co-located, off-bus) -> `detect()` -> mask +
blobs -> tracking. **No real model exists** (`find **/*.onnx` empty; `data/models/` holds only
`.gitkeep`); `OnnxDetector.detect` has never run (its only test asserts the absent-SDK
ImportError). It is **detection, not segmentation** -- `extract_blobs` returns bounding
boxes/centroids and `OnnxDetector` hard-assumes a single-channel `(1,1,H,W)` mask. No
provenance (`model_version` defaults to `"unknown"`; no hash; `MODEL_CORRUPT`/`INFERENCE_TIMEOUT`
defined but never raised). `model_path` is a `.pt` fed to onnxruntime. `tools/` is empty; a real
torch U-Net/ResNet-34 stack exists only in legacy `src/pact/model`, **no ONNX export code exists
anywhere**, and quantize is a Linear-only no-op for a conv net.
**Final-real needs.** Migrate training/eval/dataset/augment into `tools` (+ heavy deps); build
the missing PyTorch->ONNX export producing a validated frozen artifact; decide detection-only vs
true (multi-class) segmentation and fix the output contract; insert demosaic + real radiometric/
normalization so train-time and flight-time domains match; model provenance (version+hash+load
validation); latency-budget fault; real quantization with an eval-gate; model-upload/deploy
lifecycle; behavioral inference tests.

### 4.4 Sensor (monochrome + 2x2 Bayer/mosaic)
**Current.** `ImagingSensor.acquire_frame -> RawFrameMsg.raw_bands (C,H,W)` -- the contract
**presupposes already-separated bands** and exposes no raw mosaic plane and no demosaic hook.
`RealSensor` is a stub (never opens the camera; `acquire_frame -> CAMERA_STALL`; control calls
are no-ops). `SimSensor` replays zeroed `(4,256,256)` pre-separated stacks; the plume is injected
downstream via the scripted mask, so mosaic ingest/band-separation/calibration/quality are never
exercised on signal. Preprocess: identity calibration (zero dark/unit flat, never loaded from
disk), `select_bands` fancy-indexes `B2/B3/B4/B8 -> 0..3` (Sentinel-2 framing; TODO references a
"filter wheel"), quality thresholds assume normalized `[0,1]` DN that no code produces,
`MOTION_SMEAR` is an exposure-only placeholder, ROI crop is disabled. **No demosaic / dark-flat-
PRNU / bad-pixel / normalization anywhere.**
**Final-real needs.** Real PySpin acquisition + control plane + encoder stamping; a demosaic/
CFA-separation stage (decide: in driver vs preprocess -- changes the contract); characterize the
real 2x2 filter and redefine the band vocabulary; real per-pixel dark/flat/PRNU/bad-pixel
calibration + a normalization/radiance stage; physically-grounded quality models; driver
thread-safety; a raw-mosaic sim scene; real-driver tests.

---

## 5. Cross-cutting gaps (not owned by one subsystem)

- **SAFE actuation**: nobody subscribes to `ModeChangeMsg`; SAFE neither stows the gimbal nor
  quiesces the payload nor closes a cover. Enter/leave-SAFE is effectively unimplemented in
  flight.
- **Downlink + storage**: no producer of `DownlinkItemMsg`; `StorageWriteMsg`/`StorageConfig`
  and `CommsConfig` are loaded/defined but consumed by nothing; no priority queue, CRC,
  retention, or budget enforcement.
- **Command path**: no router/handlers, no ack/NACK, no seq dedup/ordering, no auth, no
  safety-inhibit ownership.
- **Model lifecycle**: no train->export->upload->stage->activate->rollback chain; `tools/` empty;
  `UploadChunkMsg`/`ModelDeployState` orphaned.
- **Config integrity**: `_validate()` is a no-op; `.get(key, default)` swallows typo'd keys;
  `main()` ignores the override path so `flight.toml` is never applied in production.
- **Process robustness**: only `KeyboardInterrupt` triggers `stop()` (no SIGTERM); no thread
  supervision/restart; FDIR app emits no heartbeat and is unmonitored (single point of failure).
- **Message hygiene**: no schema/version field; bus delivers shared (uncopied) object refs;
  default-unbounded queues with no backpressure/drop policy; large arrays typed `object`.
- **Requirements baseline**: spec Sec 12 calls for `docs/requirements/`; it does not exist. Only
  ~scattered `REQ-*` IDs in docstrings/plans -- no consolidated, traceable requirement set.
- **Doc/scope drift**: docstrings still cite dropped premises (Rust structs, multiprocessing
  transport, `pact.preprocessing.*`); spec Sec 5 (permanent `multiprocessing.Queue`,
  process-per-app) contradicts the implemented in-process thread-per-app + `queue.Queue`
  (architecture.md / ADR-0003) -- the process/transport model is not settled.
- **Legacy retirement**: `src/pact/` still present; CI gates scoped to `packages/` only; the
  root `tests/.../e2e` smoke test still targets the legacy multiprocessing code.

---

## 6. Delta, sized

Effort: S (hours), M (days), L (1-2 wk), XL (multi-week / hardware- or decision-gated).

### Sensor / ML data path
| Item | Current | Required | Effort |
|------|---------|----------|:------:|
| Demosaic / CFA separation | absent | mosaic-plane -> registered bands (driver vs preprocess) | L |
| Band vocabulary vs physical filter | Sentinel-2 B2/B3/B4/B8 placeholder | characterized real bands | L |
| Radiometric calibration + normalization | identity, no disk load | dark/flat/PRNU/bad-pixel + normalize | M |
| Quality heuristics | DN-assuming placeholders | physical smear/illumination models | M |
| RealSensor acquisition + control | `CAMERA_STALL` stub | full PySpin acquire + node-map control | L |
| Trained `.onnx` artifact | none | validated frozen artifact | XL |
| tools training + ONNX export + quantize | empty / no export / no-op | migrated stack + export + real INT8 + eval-gate | XL |
| Segmentation vs detection | blob-detection only | decide + (re)contract output | L |
| Model provenance / timeout faults | "unknown" / unraised | version+hash+validate; emit `INFERENCE_TIMEOUT` | S |
| Artifact format contract | `.pt` to onnxruntime | `.onnx` defaults | S |

### Gimbal / pointing
| Item | Current | Required | Effort |
|------|---------|----------|:------:|
| RealGimbal driver | drops cmds, reads origin | serial/CAN + encoder feedback + faults | XL |
| Command model | delta-only | absolute/rate + limits + homing/stow | M |
| Boresight-relative error + optics | absolute centroid * 0.04 scalar | frame-center error + intrinsics/LoS transform + setpoint | M |
| Safety gates wiring | implemented, not called | wire deadband/runaway/slew-limit + strikes + hysteresis | M |
| Closed-loop + SAFE stow | open-loop; SAFE no stow | encoder divergence + stow on SAFE | L |
| Gimbal/orbit dynamics twin | empty | actuator + orbit/nadir geometry | XL |

### ISS / comms / FDIR
| Item | Current | Required | Effort |
|------|---------|----------|:------:|
| RealStationLink transport | inert | bound avionics interface + framing + AOS/LOS | XL |
| Command validation + safety-inhibit + auth | none | full ingress gate | L |
| Command router + handlers + ack/NACK | half-wired (ack-only) | router + acting handlers + ack | L |
| Downlink product pipeline | no producer | producers + priority queue + CRC + budget | XL |
| On-board storage subsystem | none | persistence + retention + checksum | L |
| Model-upload + deploy lifecycle | orphaned types | uplink subsystem + reassembly + stage/rollback | L |
| SAFE actuation (cross-subsystem) | `ModeChangeMsg` unconsumed | stow/quiesce/cover + recovery | L |
| Fault persistence + ground annunciation | in-memory, discarded | NV ledger + downlink | M |
| FDIR self-monitoring / supervision | unmonitored | independent/hw watchdog + thread restart | M |

### Housekeeping / mechanical / platform
| Item | Current | Required | Effort |
|------|---------|----------|:------:|
| RealScalarSensor | `0.0` stub | real per-channel HK reads | M |
| Thermal control | monitor-only | heater HAL + control loop + survival limit | L |
| Electrical / EPS | monitor-only | conditioning + load-shed + per-rail | L |
| mechanical subsystem | empty | covers/deployables/latches + safety inhibits | XL |
| Config validation | no-op | range/typo validation -> Err | S |
| Startup/shutdown robustness | KeyboardInterrupt-only | SIGTERM + ordered bring-up/teardown + health gate | M |
| SIL fidelity | zeroed/fixed/empty-twin | raw-mosaic scenes + plume dynamics + plant/link models | XL |
| Legacy retirement + CI widen | pending | retire `src/pact`; widen gates | M |

---

## 7. Open questions for brainstorming (inputs to the spec + ADR)

These are decisions the baseline cannot settle; they shape the required-final-state.

1. **Sensor optics**: actual 2x2 mosaic tile layout + per-cell spectral response; are
   B2/B3/B4/B8 even the right physical bands? Where does demosaic live (driver vs preprocess --
   it changes the HAL contract)? Pre- or post-demosaic calibration?
2. **Model domain**: detection-only or true segmentation (and multi-class?); what radiometric
   domain (raw DN / radiance / reflectance) does the flight model expect; how to bridge
   Sentinel-2-trained weights to raw mosaic frames; flight compute target (Jetson?) -> quant
   path + latency budget.
3. **ISS avionics**: the actual facility data interface (Ethernet / serial / 1553), and whether
   the station presents CCSDS at that seam or a facility protocol; where AOS/LOS + framing live.
4. **Command + safety ownership**: which app owns auth/authorization and the hazardous-command
   inhibit gate; the command set (`command_id`/params per target) and ack/NACK contract;
   envelope versioning.
5. **SAFE semantics**: what "safe configuration" physically means (gimbal stow, cover close,
   stop-emit), the latch/debounce, and the ground-commanded recovery path.
6. **Gimbal command model**: absolute vs rate vs delta; real angle map; runaway ownership;
   stow/inhibit policy; hardware + slew/accel limits.
7. **Data system**: is on-board storage in scope; downlink prioritization + comms-budget policy;
   model-upload integrity/signing + staged-deploy acceptance.
8. **Process/transport model**: keep in-process threads or move to processes (resolve the
   spec-vs-implementation contradiction); backpressure/drop policy.
9. **Requirements**: build the consolidated `docs/requirements/` set + traceability the spec
   Sec 12 mandates.

---

## 8. Recommended next step

Brainstorm the **required-final-state requirement set** interface-by-interface (Section 7 is the
agenda), resolving the open questions into decisions. That output, diffed against this baseline,
becomes the spec (what/how/where) and the ADR(s) (why) -- including likely new ADRs for the
sensor/demosaic contract, the segmentation decision, the ISS command/safety/downlink model, the
gimbal command + closed-loop model, and the storage/model-upload subsystems.
