# PACT Software Architecture

## System Overview

PACT (Plume Autonomous Capture Technology) autonomously detects industrial smoke-stack plumes
in multispectral VNIR imagery, drives an active gimbal to track detected plumes, stores
imagery and metadata with integrity checksums, and downlinks data for ground-based ML
retraining -- all with no real-time ground-in-the-loop control.

The software is organized into ten subsystems, each running as an independent process or
thread. All inter-subsystem data passes through typed, frozen-dataclass messages on named
queues. The codebase is Python written in a Rust-idiomatic style (`frozen=True` dataclasses,
`enum.Enum` discriminants, `Result[T,E]` error handling) to enable mechanical translation
to Rust once behavior is verified.

**Phase I scope:** pretrained model inference, gimbal control, on-board storage, CCSDS
downlink, safe model uplink with rollback, and fault detection. Phase II items are called
out explicitly throughout.

---

## Mission Context and Platform Constraints

PACT is an active experiment hosted on a **TAMU-SPIRIT Pallet Carrier (PC)**, an external
ISS payload platform operated jointly by Texas A&M University and Aegis Aerospace Inc.
The PC is robotically installed on the TAMU-SPIRIT Flight Facility (TAMU-SPIRIT-FF) on the
ISS truss, operated from the Aegis Aerospace Payload Operations Control Center (POCC) in
Houston. All software and hardware design decisions are bounded by the constraints below.

### TAMU-SPIRIT PC Hardware Envelope

The PACT experiment must fit within the PC integration volume. The PC integration plate
uses a 3" × 2" hole pattern, threaded 8-32, for mechanical attachment. A removable cover
protects the experiment during launch; the crew removes it on-orbit before deployment.

| Parameter | Value | Source |
|-----------|-------|--------|
| PC volume (L × W × H) | 421.64 × 218.44 × 250.69 mm (16.60" × 8.60" × 9.87") | TAMU-SPIRIT QRG §2.1.2 |
| PC mass limit | ≤ 20 lbs (≈ 9.1 kg) per PC | TAMU-SPIRIT QRG Table 6 |
| PACT mass target (should) | ≤ 5 kg | REQ-RESO-HIGH-005 |
| PACT mass limit (shall) | ≤ 20 kg | REQ-RESO-HIGH-004 |

### Power Interface

Power is supplied by the TAMU-SPIRIT-FF through connectors on the PC mounting plate.
Power may be removed from the experiment at any time without warning — the software must
handle unexpected power loss gracefully (REQ-SAFE-HIGH-002).

| Parameter | Value | Source |
|-----------|-------|--------|
| Power limit (shall) | ≤ 60 W total | TAMU-SPIRIT QRG §2.2.1 / REQ-RESO-HIGH-001 |
| Power target (should) | ≤ 15–20 W | Introductory meeting notes |
| Input voltage options | Fixed 28 V or variable 3.3–20 V | TAMU-SPIRIT QRG §2.2.1 |
| Max current at 28 V | 2.1 A | TAMU-SPIRIT QRG §2.2.1 |
| Max current (variable) | 1 A per output | TAMU-SPIRIT QRG §2.2.1 |

The Jetson Xavier NX is powered via the ISS 28 V bus through the Vicor V28C12C100BL
DC-DC converter (28 V → 12 V, 100 W, ~90% efficiency) documented in the imaging system
BOM. The `[fault]` `power_limit_w = 55.0` setting in `config/default.toml` provides a
5 W margin below the 60 W hard cap.

### Data and Communications Interface

All PACT data travels through the TAMU-SPIRIT-FF and is embedded in the ISS downlink,
routed via the Aegis POCC before reaching the PACT ground team. All uplink commands are
processed **manually** by TAMU-SPIRIT operators — no autonomous ground commanding.

| Parameter | Value | Source |
|-----------|-------|--------|
| Physical interface | RS-422 or Ethernet 10/100 | TAMU-SPIRIT QRG Table 2 |
| RS-422 max data rate | 10–20 Mbps | TAMU-SPIRIT QRG Table 2 |
| Ethernet sustained throughput | ~4 MB/s | TAMU-SPIRIT QRG Table 2 |
| Max downlink per weekday | 1 GB | TAMU-SPIRIT QRG §2.2.2 / REQ-COMM-HIGH-002 |
| Max uplink per weekday | 100 MB | TAMU-SPIRIT QRG §2.2.2 / REQ-COMM-HIGH-002 |
| Max downlink rate | 5 Mbps | REQ-COMM-HIGH-001 |
| Max uplink rate | 2 Mbps | REQ-COMM-HIGH-001 |
| Comm window | Weekdays only (MON–FRI) | TAMU-SPIRIT QRG §2.2.2 |
| Operator hours | Up to 5 hrs/week operations; 10 hrs/month anomaly resolution | TAMU-SPIRIT QRG Table 6 |

The weekday-only comm window is not a software approximation — it reflects the
TAMU-SPIRIT operational schedule enforced by Aegis Aerospace operators. The software
enforces this as a UTC weekday gate in `comms/`.

### Structural and Thermal Environment

Experiments must pass NASA 7000 GEVS standard qualification. Fasteners require Loctite
222MS with torque specification. All components must be thermally rated for both ground
testing and on-orbit conditions.

| Parameter | Value | Source |
|-----------|-------|--------|
| Launch vibration (random) | 4.7 Grms axial and lateral (per MISSE-MSC spec) | TAMU-SPIRIT QRG Fig. 10 / REQ-STRC-HIGH-002 |
| Launch quasi-static load | ≤ 9.3 G axial and lateral | REQ-STRC-HIGH-001 |
| Ground thermal test range | −46°C to +55°C (−50°F to 130°F) | TAMU-SPIRIT QRG §2.1.3 |
| On-orbit operational range | −40°C to +85°C | REQ-THRM-HIGH-001 |
| Vibration test axes | All three (X, Y, Z), hard-mounted | TAMU-SPIRIT QRG §2.1.3 |

**Implication for gimbal selection:** Most COTS FPV gimbals are rated to 0–50°C only,
which does not meet REQ-THRM-HIGH-001. The COBRA-HPX (-35°C to +70°C operating) is the
only trade study candidate close to meeting this requirement; however, it still falls short
of the −40°C lower bound and would require thermal qualification testing.

### Safety and Compliance

- Must comply with ISS external payload safety requirements; Aegis Aerospace obtains NASA
  safety approval as part of their standard service (REQ-SAFE-HIGH-001)
- Must provide fail-safe behavior on fault, power loss, or loss of command (REQ-SAFE-HIGH-002)
- Must meet ISS EMI/EMC requirements per SSP 30237 (REQ-ELEC-HIGH-001)
- Materials must be ISS-compatible: no outgassing above NASA low-outgassing limits,
  no contamination risk to adjacent experiments (REQ-SAFE-HIGH-003)
- Cameras and sensors on the TAMU-SPIRIT-FF may require a NOAA and/or FCC license and
  a NASA viewing analysis (TAMU-SPIRIT QRG §2.3.2)

### Mission Goals Traceability

The following goals from `PACT_Needs_and_Goals.docx` are in scope for Phase I and
are the primary traceability targets for all requirements in this document:

| Goal ID | Summary | Phase |
|---------|---------|-------|
| GOAL-001 | Acquire high spatiotemporal fidelity plume imagery | Phase I |
| GOAL-002 | Close the gimbal autonomy loop with SOTA inference | Phase I |
| GOAL-003 | Collect complete training metadata for ML reuse | Phase I |
| GOAL-004 | Ensure dataset usability and integrity | Phase I |
| GOAL-005 | Operate within resource and ops constraints | Phase I |
| GOAL-006 | Maintain operational reliability with weekly monitoring | Phase I |
| GOAL-008 | Safe model deployment lifecycle (uplink + rollback) | Phase II |
| GOAL-011 | Iterate model update cycle if resources allow | Phase II |

Goals GOAL-007, GOAL-009, and GOAL-010 are explicitly **not in scope**.

---

## Subsystems

| Subsystem        | Role                                                                       |
|------------------|----------------------------------------------------------------------------|
| **types**        | Dependency root: all enums, frozen message dataclasses, config dataclasses |
| **model**        | U-Net/ResNet-34 segmentation -- forward pass, blob extraction, training    |
| **preprocessing**| Band selection, radiometric calibration, quality flagging, ROI crop        |
| **controller**   | Gimbal arbiter state machine, blob tracker, EMA filter, Kalman + LQR      |
| **imaging**      | FLIR Blackfly S GigE Vision interface, frame acquisition, stall detection  |
| **comms**        | CCSDS encoding, priority downlink queue, chunked uplink, staged model deploy|
| **storage**      | Frame persistence with SHA-256 checksums and append-only manifests         |
| **telemetry**    | Health event aggregation and CCSDS telemetry packet formatting              |
| **fault**        | Heartbeat watchdog, fault detection, safe-mode entry                       |
| **ops**          | Process orchestrator: queue creation, config loading, mode FSM             |

---

## Process Topology

All queues are created in `ops/main.py` and passed as arguments to each subsystem.
No subsystem creates its own queues.

```
imaging_process        --[raw_frame_queue: RawFrameMsg]-->        inference_process
                                                                   (preprocessing runs here)
inference_process      --[inference_queue: InferenceResultMsg]--> controller_process
controller_process     --[gimbal_queue: GimbalCommandMsg]-->       (hardware gimbal stub)
controller_process     --[telemetry_queue: TelemetryEventMsg]-->   telemetry_process
inference_process      --[storage_queue: StorageWriteMsg]-->       storage_process
storage_process        --[downlink_queue: DownlinkItemMsg]-->       comms_process
telemetry_process      --[downlink_queue: DownlinkItemMsg]-->       comms_process
comms_process          --[uplink_queue: UploadChunkMsg]-->          ops/main.py
any subsystem          --[fault_queue: FaultEventMsg]-->            fault_process
any subsystem          --[heartbeat_queue: HeartbeatMsg]-->         fault_process
fault_process          --[mode_queue: ModeChangeMsg]-->             ops/main.py
```

**Preprocessing co-location:** preprocessing runs as a plain function call inside the
inference process (`_run_inference_process()` in `ops/main.py`), not as a separate process.
This avoids serializing large numpy arrays over a `multiprocessing.Queue` on the hot path.
The `RawFrameMsg -> ProcessedFrameMsg` transformation is a function call, not a queue hop.

---

## Concurrency Model

| Subsystem   | Primitive                          | Rationale                                               |
|-------------|------------------------------------|---------------------------------------------------------|
| imaging     | `threading.Thread` + `queue.Queue` | I/O-bound GigE Vision DMA; GIL not a bottleneck         |
| inference   | `multiprocessing.Process`          | GPU-bound; requires true process isolation (REQ-AIML-COMP-002) |
| controller  | `multiprocessing.Process`          | CPU-bound; isolated from GIL for deterministic scheduling |
| storage     | `threading.Thread` + `queue.Queue` | I/O-bound disk writes                                   |
| comms       | `asyncio`                          | Many concurrent I/O waiters; no CPU-heavy loops         |
| telemetry   | `threading.Thread` + `queue.Queue` | I/O-bound serialization and queue puts                  |
| fault       | `multiprocessing.Process`          | GIL immunity: watchdog must fire during GPU kernels     |
| ops         | Main process                       | Spawns all others; runs mode FSM loop                   |

---

## Dependency Layer Order

The import graph is strictly layered. Lower layers must never import from higher layers;
a circular import is a build-breaking bug.

```
types                           <- dependency root; imports only Python stdlib
  |
  v
model / preprocessing / imaging <- import from types only
  |
  v
controller                      <- imports from types + preprocessing
  |
  v
storage / telemetry / comms     <- import from types + model
  |
  v
fault                           <- imports from types + all subsystem message types
  |
  v
ops                             <- imports everything; orchestration layer
```

---

## Imaging Subsystem — Requirements and Constraints

The imaging subsystem (`src/pact/imaging/`) acquires raw multispectral frames from the
FLIR Blackfly S GigE 5MP camera and places them on `raw_frame_queue` as `RawFrameMsg`.
The following requirements from `Imaging_Mid-Level_Requirements.xlsx` directly constrain
what the software must do.

### Optical and Sensor Parameters (Fixed Hardware — Not Configurable)

These are physical facts about the selected camera + lens + filter combination. They are
not tunable in software but every subsystem must be designed around them.

| Parameter | Value | Derivation |
|-----------|-------|------------|
| ISS orbital altitude | 420 km | Standard ISS operating altitude |
| ISS ground track velocity | ~7.7 km/s | Orbital mechanics |
| Sensor pixel pitch | 3.45 µm | FLIR Blackfly S GigE 5MP (Sony IMX264) |
| Focal length | 150 mm | Selected lens (Nikon 300mm f/4E at effective focal length via crop) |
| Ground sampling distance (GSD) | ≤ 10 m | ACTP-IMAG-001.001: GSD = pixel_pitch × altitude / focal_length |
| Sensor array | 2448 × 2048 px | FLIR Blackfly S GigE 5MP |
| Swath width (cross-track) | ≥ 23.6 km | ACTP-IMAG-001.005 |
| Swath width (along-track) | ≥ 19.8 km | ACTP-IMAG-001.005 |
| Spectral range | 490 nm – 842 nm (4 bands: 490, 560, 665, 842 nm) | ACTP-IMAG-001.002 |
| Diffraction limit | ≤ 12 µm @ 842 nm, ≤ 7 µm @ 490 nm | ACTP-IMAG-001.002 |

### Exposure Constraint (Software-Enforced — Critical)

**Maximum exposure time: ≤ 1.25 ms** (ACTP-IMAG-001.008)

At ISS ground track velocity of 7.7 km/s, each millisecond of exposure smears the image
by 7.7 m on the ground. The requirement is to limit motion blur to ≤ 1 GSD (≤ 9.66 m),
which gives the 1.25 ms ceiling. The camera driver in `imaging/camera.py` must enforce
this limit on exposure configuration and must never allow a user-supplied config to exceed
it. Any frame acquired with exposure > 1.25 ms must be flagged with `MOTION_SMEAR` in the
quality flags by `preprocessing/quality.py`.

### Acquisition Modes (Software State Machine Requirement)

The system must support two distinct acquisition modes (ACTP-IMAG-001.007), which map
directly to the Ops Mode FSM states in `ops/main.py`:

| Mode | FSM State | Behavior |
|------|-----------|----------|
| **Survey/Scan** | `SCAN` | Wide-area nadir imaging at low slew rate (≤ 0.5 deg/s) to discover plume events |
| **Tracking/Burst** | `ACTIVE` / `TRACKING` | High-cadence imaging locked to detected plume ROI; gimbal actively tracking |

The transition from SCAN → TRACKING is the primary closed-loop autonomy demonstration
(GOAL-002). Frames acquired in SCAN mode are usability-tagged `TRAINING`; frames in
TRACKING/burst mode are tagged `TRAINING` + `HIGH_VALUE`.

### Metadata Requirements (Storage and Comms Impact)

Every stored frame must include the following metadata fields (ACTP-IMAG-002.*).
These are enforced by `storage/writer.py` and verified by `fault/detector.py`:

| Field | Requirement | Format | Req ID |
|-------|-------------|--------|--------|
| Geolocation | Latitude + longitude per frame | JSON | ACTP-IMAG-002.001 |
| Timestamp | ± 100 ms precision, GPS-synchronized, UTC | ISO 8601 string | ACTP-IMAG-002.002 |
| Metadata format | Machine-readable | JSON (not XML — see `storage/writer.py`) | ACTP-IMAG-002.003 |
| File manifest | Per-session manifest listing all image files | JSON lines | ACTP-IMAG-003.003 |
| Config version | Active software + sensor settings version | string | ACTP-IMAG-003.004 |

**Note on timestamp precision:** the ± 100 ms figure is a rough estimate per the
requirements spreadsheet and is flagged for refinement. GPS sync implementation is TBD —
currently `timestamp_utc` in `RawFrameMsg` is set by the Jetson system clock, which may
drift. A GPS disciplined clock or NTP sync to ISS time must be validated before flight.

### Pre-Flight Verification Requirements

These are ground test gates that must be passed before the imaging system is flight-ready.
They are not software tests but they drive what the software must be able to demonstrate:

| Verification | What it Requires | Req ID |
|-------------|-----------------|--------|
| Spectral performance | Software must be able to select and log which spectral bands are active; contrast metric must be computable from captured data | ACTP-IMAG-004.001 |
| Mechanical integration | C-Mount interface verified; no software dependency | ACTP-IMAG-004.002 |
| End-to-end functional test | Full pipeline (camera → preprocessing → inference → storage) must run on ground hardware | ACTP-IMAG-004.003 |
| Environmental compatibility | COTS camera and lens must pass thermal/vibe analysis; `MockCamera` must be usable as drop-in replacement for all CI | ACTP-IMAG-004.004 |

---

## Fault Detection and Risk Mitigations

This section documents the software-level mitigations for risks identified in
`PACT_Risk_Analysis.xlsx` and `Overall_Fault_Mitigation.xlsx`. These drive concrete
implementation requirements in `fault/detector.py` and across all subsystems.

### Gimbal Risks → Software Mitigations

| Risk ID | Event | Software Mitigation | Where Implemented |
|---------|-------|--------------------|--------------------|
| RISK-GIMB-001 | Encoder bias / calibration error → pointing inaccuracy | Kalman filter absorbs slow bias drift; boresight calibration routine run at startup; encoder readings validated against commanded position each loop cycle | `controller/kalman.py`, `controller/process.py` |
| RISK-GIMB-002 | Vibration / flexible mode coupling → jitter exceeds limit | LQR gain tuning to suppress resonant frequencies; jitter monitored via RMS of pointing error in telemetry; `GIMBAL_JITTER` fault emitted if RMS > threshold | `controller/lqr.py`, `fault/detector.py` |
| RISK-GIMB-003 | Insufficient torque/slew rate → missed ROI onset | `max_slew_rate_deg_per_s` enforced as soft ceiling; if slew command cannot be completed within `settling_time` threshold, emit `GIMBAL_SLOW` telemetry warning | `controller/process.py` |
| RISK-GIMB-004 | Gimbal power instability → imaging interruption | Voltage monitoring via `fault/detector.py` `check_power()`; graceful shutdown on voltage drop; `POWER_FAULT` emitted; gimbal parked to safe position before power removed | `fault/detector.py`, `ops/main.py` |
| RISK-GIMB-005 | Encoder signal drop / corruption → position feedback lost | REQ-CTRL-FAULT-001: detect loss within TBD cycles → halt actuator, hold position, transition to SAFE; cross-reference with motor current if available | `controller/process.py`, `fault/detector.py` |

### Model / Inference Risks → Software Mitigations

| Risk ID | Event | Software Mitigation | Where Implemented |
|---------|-------|--------------------|--------------------|
| RISK-AIML-001 | Domain shift (on-orbit vs training distribution) → degraded detection | Confidence gate (0.55) filters low-confidence outputs; out-of-distribution frames tagged in quality flags; low-confidence outputs prohibited from aggressive retargeting | `controller/process.py`, `preprocessing/quality.py` |
| RISK-AIML-002 | Partial/distorted plume visibility → mis-detection | Image quality gates in `preprocessing/quality.py`: saturation, sun-glint, NIR/red ratio; frames failing gates tagged and downlinked separately from tracking-quality frames | `preprocessing/quality.py` |
| RISK-AIML-003 | Detection confidence oscillation → gimbal chatter | Persistence gates (`acquire_persistence_frames = 3`, `release_persistence_frames = 5`); EMA smoothing on centroid; deadband suppresses small displacements; rate limiter caps command frequency | `controller/arbiter.py`, `controller/tracker.py` |
| RISK-AIML-004 | Image artifacts (blur, rolling shutter) → model noise | Exposure ≤ 1.25 ms enforced; `MOTION_SMEAR` quality flag; frames with severe blur down-weighted in `FrameUsabilityTag` | `imaging/camera.py`, `preprocessing/quality.py` |
| RISK-AIML-005 | Compute throughput drop → inference latency spike | Inference isolated in `multiprocessing.Process`; `latency_budget_ms` watchdog returns `Err(INFERENCE_TIMEOUT)`; degraded mode (lower frame rate, ROI crop) planned for Phase II | `model/engine.py`, `fault/detector.py` |
| RISK-AIML-006 | Radiation bit flip / numerical instability → corrupted output | NaN/Inf/out-of-range output sanity checks before passing to arbiter; model file SHA-256 verified at load; runtime watchdog via heartbeat system; rollback model always retained | `model/engine.py`, `comms/uplink.py`, `fault/detector.py` |

### Communications Risks → Software Mitigations

| Risk ID | Event | Software Mitigation | Where Implemented |
|---------|-------|--------------------|--------------------|
| RISK-COMM-001 | Instantaneous rate exceeds 5 Mbps / 2 Mbps | Hardware rate limiter in `comms/`; `max_downlink_rate_bps = 5000000` and `max_uplink_rate_bps = 2000000` enforced in downlink queue | `comms/downlink.py` |
| RISK-COMM-002 | Daily budget exhausted before end of day | Onboard daily byte counter; `max_daily_downlink_bytes = 1 GB`, `max_daily_uplink_bytes = 100 MB`; counter resets at UTC midnight; `DOWNLINK_BUDGET_EXHAUSTED` fault emitted | `comms/downlink.py`, `fault/detector.py` |
| RISK-COMM-003 | Model upload interrupted mid-transfer | Chunked transfer protocol with per-chunk acknowledgement; configurable `uplink_reassembly_timeout_s` (Phase I gap — see TODO.md); incomplete uploads emit `MODEL_CORRUPT` | `comms/uplink.py` |
| RISK-COMM-004 | Corrupted model activated without verification | Model staged to `data/models/staged.pt`; SHA-256 verified before activation; smoke test run before switching active model; rollback model always retained at `data/models/rollback.pt` | `comms/uplink.py`, `ops/main.py` |

### Electrical Risks → Software Mitigations

| Risk ID | Event | Software Mitigation | Where Implemented |
|---------|-------|--------------------|--------------------|
| RISK-ELEC-001 | Power spike damages experiment | EMI input filter (hardware); software detects voltage anomaly via `check_power()`; graceful shutdown saves state and parks gimbal before power is cut | `fault/detector.py`, `ops/main.py` |
| RISK-ELEC-002 | Power loss (ISS can remove power without warning) | All critical state is persisted to `storage/` before each write completes; SHA-256 checksums allow detection of incomplete writes on restart; gimbal parked to safe position on power loss detection | `storage/writer.py`, `fault/detector.py` |

### Thermal Risks → Software Mitigations

| Risk ID | Event | Software Mitigation | Where Implemented |
|---------|-------|--------------------|--------------------|
| RISK-THRM-001 | Thermal exposure damages optical components | TEC module (hardware) actively controls sensor temperature; software monitors TEC status; `check_thermal()` emits fault if sensor temperature exceeds `thermal_limit_c = 80.0` | `fault/detector.py` (HAL TBD — Phase II) |
| RISK-THRM-002 | Thermal damage to gimbal joints | Motor temperature monitoring per REQ-CTRL-FAULT-003; two-stage response (warn at TBD°C, halt at TBD°C); HAL TBD pending hardware selection | `controller/process.py` (TBD) |
| RISK-THRM-003 | Jetson Xavier overheats or underheats | INA3221 power/thermal sensor polling (Phase II); `thermal_limit_c = 80.0` in `[fault]` config; Jetson thermal throttle detection via inference latency spike | `fault/detector.py` (HAL TBD — Phase II) |
| RISK-THRM-004 | Cooling system failure / fluid contamination | Passive thermal design preferred; no active fluid cooling in PACT; TEC module is solid-state with no fluid risk | N/A |

### Fault Detection Methods (from Overall_Fault_Mitigation.xlsx)

The following detection methods are the **primary** implementation targets for
`fault/detector.py`. Secondary and optional methods are deferred to Phase II.

| Subsystem | Fault | Detection Method | Priority | Notes |
|-----------|-------|-----------------|----------|-------|
| Gimbal | Tracking error / sensor drift | Command vs. feedback check (compare commanded to encoder position each loop) | Primary | Cross-reference with motor current once HAL available |
| Gimbal | Runaway motion | Rate/acceleration limit check | Primary | Already enforced via `max_slew_rate_deg_per_s` |
| Gimbal | Stiction / overload | Motor current monitoring | Primary | HAL TBD |
| Gimbal | Stall / control freeze | Motion timeout watchdog | Primary | REQ-CTRL-FAULT-002 |
| Model | Corrupted inputs | Input sanitization (NaN, Inf, out-of-range check) | Primary | Before every `InferenceEngine.run()` call |
| Model | Invalid predictions | Output sanity checks (impossible bbox, confidence out of [0,1]) | Primary | Before passing to arbiter |
| Model | Frozen inference loop | Runtime watchdog via heartbeat system | Primary | Already implemented via heartbeat contract |
| Model | Weight corruption | Model hash verification (SHA-256 at load) | Secondary | Already implemented in `comms/uplink.py` |
| Comms | Bit errors | CRC/checksum on all packets | Primary | CCSDS CRC-32 implemented; CRC-16/CCITT deferred |
| Comms | Packet loss / reorder | Sequence numbers | Primary | TBD in CCSDS secondary header (Phase II) |
| Comms | Link dropout | Heartbeat/keepalive | Primary | Already implemented via heartbeat contract |
| Comms | Stale data | Telemetry freshness check (timestamp age) | Primary | TBD in `fault/detector.py` |
| Imaging | Corrupted frames | Frame CRC/checksum | Primary | SHA-256 per frame in `storage/writer.py` |
| Imaging | Black / saturated frames | Histogram / pixel stats | Primary | Partial — saturation check in `preprocessing/quality.py` |
| Imaging | Frozen / repeated frames | Frame difference check | Primary | TBD in `preprocessing/quality.py` |
| All | Power loss / undervoltage | Voltage monitoring + graceful shutdown | Primary | HAL TBD — Phase II via INA3221 |

---

## Controller Subsystem — Detailed Design

### Overview

The controller subsystem is owned by Vin Manoj Nair. It is responsible for all gimbal
pointing logic: receiving `InferenceResultMsg` outputs, filtering blobs through safety
gates, running the gimbal state machine, computing LQR control commands, and emitting
`GimbalCommandMsg` to the hardware interface. It runs as an independent
`multiprocessing.Process` (see Concurrency Model) to ensure deterministic scheduling
independent of GPU kernel timing in the inference process.

The subsystem is organized into four layers:

```
InferenceResultMsg (from inference_queue)
        |
        v
  [Safety Gates]          -- process.py: confidence, area, deadband, rate-limit
        |
        v
  [GimbalArbiter]         -- arbiter.py: pure-function state machine (IDLE/ACQUIRING/TRACKING/SCAN/SAFE)
        |
        v
  [Blob Tracker + EMA]    -- tracker.py: IoU blob matching + EMA centroid smoothing
        |
        v
  [Kalman Filter]         -- kalman.py: state estimator (position + rate per axis)
        |
        v
  [LQR Controller]        -- lqr.py: optimal control law, gain from offline DARE solve
        |
        v
  GimbalCommandMsg (to gimbal_queue) + TelemetryEventMsg (to telemetry_queue)
```

---

### Control Architecture (REQ-CTRL-ARCH-001 through REQ-CTRL-ARCH-004)

#### Linearized State-Space Model

The gimbal is modeled as two decoupled single-axis systems (azimuth and elevation).
Each axis has two states: angular position θ and angular rate θ̇. The full state vector is:

```
x = [θ_az, θ̇_az, θ_el, θ̇_el]ᵀ     (4 × 1)
u = [u_az, u_el]ᵀ                    (2 × 1, normalized motor commands)
```

The continuous-time linearized dynamics per axis are:

```
ẋ = A·x + B·u
y = C·x

A = [[0, 1],       B = [[0  ],      C = [[1, 0]]
     [0, 0]]            [1/J]]
```

where J is the effective rotational inertia of the gimbal + payload. The discrete-time
model is obtained via zero-order hold at the control loop sample period `dt = kalman_dt_s`
(default 0.1 s). Gains are computed **offline** using the DARE (Discrete Algebraic Riccati
Equation) solver and stored as configurable parameters in `config/default.toml`. This
satisfies REQ-CTRL-ARCH-004.

**TBD:** J must be characterized once the gimbal hardware is selected (see Gimbal Hardware
Selection below). Until then, `lqr.py` falls back to proportional gain; see TODO.md for
the known bug in this fallback path.

#### LQR Gain Computation

The LQR minimizes the infinite-horizon cost:

```
J = Σ (xᵀQx + uᵀRu)
```

Weighting matrices are diagonal and configurable:

```toml
lqr_Q_diag = [10.0, 10.0, 1.0, 1.0]   # [θ_az, θ̇_az, θ_el, θ̇_el] — penalize position error heavily
lqr_R_diag = [1.0, 1.0]               # [u_az, u_el] — penalize control effort equally
```

Higher Q/R ratio → more aggressive pointing correction. Lower ratio → smoother, slower
response. Gains are re-derived offline whenever the plant model changes and flashed as
config. The DARE solver runs in `lqr.py::LqrController.from_config()` at startup.

#### Kalman Filter State Estimator (REQ-CTRL-ARCH-002)

The Kalman filter estimates gimbal angular position and rate from encoder measurements.
It runs at the control loop rate (REQ-CTRL-ARCH-003, TBD Hz) and is decoupled from the
ML inference pipeline rate. Configuration:

```toml
kalman_dt_s             = 0.1    # Sample period (must match control loop rate)
kalman_process_noise    = 0.01   # Q matrix diagonal — tuned to gimbal jitter model
kalman_measurement_noise = 0.1   # R matrix diagonal — tuned to encoder noise floor
```

The estimator implements a standard discrete-time Kalman predict/update cycle:

```
Predict:  x̂⁻ = F·x̂,   P⁻ = F·P·Fᵀ + Q
Update:   K  = P⁻·Hᵀ·(H·P⁻·Hᵀ + R)⁻¹
          x̂  = x̂⁻ + K·(z − H·x̂⁻)
          P  = (I − K·H)·P⁻
```

where z is the encoder position measurement, H = [1, 0] (position observable only).

#### Control Loop Rate (REQ-CTRL-ARCH-003)

The control loop executes at a fixed rate independent of the ML inference pipeline. The
target rate is TBD — it will be set once the gimbal hardware is selected and its servo
bandwidth is characterized. The control loop rate must be at least 2× the gimbal's
mechanical bandwidth (Nyquist). The rate is enforced using `threading.Event.wait(timeout)`
(not `time.sleep`) consistent with the heartbeat contract.

---

### Safety Gates

Applied in `controller/process.py` before the arbiter is called. All four must pass for
a blob to be considered command-eligible.

| Gate | Parameter | Default | Failure Action |
|------|-----------|---------|----------------|
| Confidence gate | `confidence_gate` | 0.55 | Blob dropped silently |
| Minimum area gate | `min_blob_area_px` | 15 px | Blob dropped silently |
| Deadband check (lower) | `min_deadband_px` | 20 px | No command issued (within deadband — acceptable jitter) |
| Deadband check (upper) | `max_deadband_px` | 250 px | `GIMBAL_RUNAWAY` fault emitted |
| Rate limiter | `retarget_rate_limit_hz` | 0.5 Hz | Command suppressed until interval elapses |

---

### Pointing Performance Requirements

These are the verified pointing requirements from `Gimball_Control_Requirements.xlsx`.
Numeric thresholds marked **(TBD)** are pending gimbal hardware selection and
characterization testing.

| Requirement ID | Parameter | Threshold | Verification |
|----------------|-----------|-----------|--------------|
| REQ-CTRL-POINT-001 | Steady-state pointing error (per axis, RMS) | **(TBD)** deg | Test |
| REQ-CTRL-POINT-002 | Settling time to within **(TBD)** deg of command | **(TBD)** s | Test |
| REQ-CTRL-POINT-003 | Jitter during steady-state tracking (per axis, RMS) | **(TBD)** deg | Test |
| REQ-CTRL-SLEW-001 | Maximum retarget slew rate (any axis) | **(TBD)** deg/s | Test |
| REQ-CTRL-SLEW-002 | Scan mode slew rate | ≤ 0.5 deg/s | Test |

Rationale: pointing accuracy directly determines whether the detected plume centroid
remains within the camera FOV for science data quality (traces to GOAL-001, GOAL-002,
REQ-GIMB-HIGH-001, REQ-GIMB-HIGH-004).

---

### Angle Limits (REQ-CTRL-LIMIT-001 through REQ-CTRL-LIMIT-003)

Two-tier software limit system implemented in `process.py`:

**Hard limits (REQ-CTRL-LIMIT-001):** Commands that would move the gimbal beyond **(TBD)**
deg from boresight on any axis are clipped to the limit value. A `limit_reached` flag is
set in telemetry. Hard limits prevent mechanical damage to the gimbal, camera cable
harness, and adjacent ISS structure.

**Soft limits (REQ-CTRL-LIMIT-002):** Configurable inner boundary set no less than **(TBD)**
deg inside the hard limits. On soft limit trigger, slew rate is reduced to ≤ **(TBD)** deg/s
and a soft-limit warning is logged. Provides a controlled deceleration buffer before the
hard stop.

**Output saturation (REQ-CTRL-LIMIT-003):** Actuator command outputs are clamped to the
valid motor driver range at all times. Saturated commands are flagged in control loop
telemetry. This is the last line of defense before the hardware interface.

---

### Fault Handling (REQ-CTRL-FAULT-001 through REQ-CTRL-FAULT-003)

| Requirement ID | Fault Condition | Detection | Response |
|----------------|----------------|-----------|----------|
| REQ-CTRL-FAULT-001 | Encoder loss | Missing feedback within **(TBD)** control loop cycles | Halt actuator commands, hold last known position, transition arbiter to SAFE state |
| REQ-CTRL-FAULT-002 | Control loop stall | Loop fails to execute within **(TBD)** ms of scheduled period | Watchdog triggers safe halt, fault logged |
| REQ-CTRL-FAULT-003 | Motor over-temperature (warn) | Temperature exceeds **(TBD)** °C | Reduce max slew rate by **(TBD)**%, log thermal warning |
| REQ-CTRL-FAULT-003 | Motor over-temperature (critical) | Temperature exceeds **(TBD)** °C | Halt all gimbal motion, enter safe hold |

All fault events are emitted as `FaultEventMsg` on `fault_queue` to `fault_process`.
The note on REQ-CTRL-FAULT-001 specifies that a SAFE state must be added to the
`GimbalArbiter` FSM (currently tracked as a known gap).

---

### Control Loop Telemetry (REQ-CTRL-TELEM-001 and REQ-CTRL-TELEM-002)

Every control loop cycle logs the following fields (emitted as `TelemetryEventMsg`):

| Field | Units | Description |
|-------|-------|-------------|
| `cmd_az`, `cmd_el` | deg | Commanded position per axis |
| `est_az`, `est_el` | deg | Kalman-estimated position per axis |
| `err_az`, `err_el` | deg | Position error per axis |
| `ctrl_az`, `ctrl_el` | normalized | LQR control output per axis |
| `loop_exec_ms` | ms | Control loop execution time |
| `saturated_az`, `saturated_el` | bool | Output saturation flags |
| `soft_limit_az`, `soft_limit_el` | bool | Soft limit flags per axis |
| `hard_limit_az`, `hard_limit_el` | bool | Hard limit flags per axis |

**Weekly downlink summary** (REQ-CTRL-TELEM-002): mean pointing error per axis, max
pointing error per axis, total retarget commands issued, total limit-reached events, and
total fault events. Included in the weekly CCSDS downlink bundle.

---

### Gimbal Hardware Selection

The gimbal has not yet been selected. The trade study (`Gimbal_Trade_Study.xlsx`) is
active. The selection directly gates several TBD thresholds above and the
`send_gimbal_command()` hardware interface implementation in `controller/process.py`
(currently a stub — see TODO.md).

**Candidates under evaluation:**

| Candidate | Mass | DOF | Slew Rate | Payload Capacity | Status | Notes |
|-----------|------|-----|-----------|-----------------|--------|-------|
| Tethers Unlimited COBRA-HPX | 184 g (276 g w/ locks) | 3 | 30 deg/s | 1200 g (zero-G) | Flight qualified | 12-bit absolute encoder, 3× FC2 frangibolts, 2.4 W, -35°C to +70°C operating |
| Tethers Unlimited COBRA-UHPX | 491 g (w/ locks) | 3 | 180 deg/s | load dependent | Qual pending | 100:1 harmonic drive, ≤3 arc-sec resolution |
| Gremsy Pixy U | 465 g | 3 | 100 deg/s | 456 g | COTS | Not space-qualified; 0–50°C op range; USB interface |
| Electric Propulsion Lab GIM-3X25-4-F | 1240 g | 3 | 2 deg/s | 4000 g | Messaged | ±25° range, M80 DataMate interface |
| C-20D 2-axis FPV Gimbal | 36 g | 2 | ±1500 deg/s | 20 g | COTS | Very low mass; camera size constraint |

**Key selection drivers:**
- Mass budget: camera (33 g) + lens (755 g) + gimbal must remain within the 5 kg
  estimated payload allocation (REQ-RESO-HIGH-005)
- Space qualification: LEO thermal environment (-40°C to +85°C per REQ-THRM-HIGH-001),
  launch loads (9.3G axial, 4.7 Grms per REQ-STRC-HIGH-001/002)
- Interface: the `send_gimbal_command()` stub in `controller/process.py` must be
  replaced with a real serial/CAN driver matched to the selected hardware
- Slew rate: must satisfy REQ-CTRL-SLEW-001 threshold once characterized
- The COBRA-HPX is the only flight-qualified candidate with documented space heritage;
  it is the current leading option pending mass and interface confirmation

**Blocking dependency:** until hardware is selected, the following remain TBD in the
codebase: hard/soft limit angles, control loop rate, LQR plant model (inertia J),
Kalman tuning, all REQ-CTRL-POINT/SLEW/LIMIT/FAULT numeric thresholds, and the
hardware driver in `controller/process.py`.

---

## State Machines

### Gimbal Arbiter (controller/)

Five states. Transitions are evaluated once per `InferenceResultMsg`. The arbiter is a
pure function -- all state is explicit in `ArbiterState`.

```
                  blob detected, confidence gate passed,
                  persistence_count < acquire_persistence_frames
IDLE ------------------------------------------------------------> ACQUIRING
  ^                                                                    |
  |   consecutive_miss >= release_persistence_frames                  | persistence_count >=
  <----------------------- TRACKING <---------------------------------+  acquire_persistence_frames
  |                             |
  |   consecutive_miss >=       | blob detected
  |   release_persistence       | (resets miss counter)
  |   frames                    |
  <-----------------------------+

IDLE ---- idle_seconds > scan_entry_idle_seconds --------------> SCAN
SCAN ---- blob detected ----------------------------------------> ACQUIRING

any state ---- FaultEventMsg received -------------------------> SAFE
SAFE ---- ground command (Phase II) ---------------------------> IDLE
```

**Guards applied before arbiter is called (in `process.py`):**
- `apply_confidence_gate`: mean blob confidence >= `confidence_gate` (default 0.55)
- `apply_min_area_gate`: blob area >= `min_blob_area_px` (default 15 px)
- `check_deadband`: displacement in `[min_deadband_px, max_deadband_px]`; > max -> `GIMBAL_RUNAWAY`
- `check_rate_limit`: enforce <= `retarget_rate_limit_hz` (default 0.5 Hz) between commands

### Ops Mode FSM (ops/)

```
IDLE ---- first blob tracked ----------------------------------> ACTIVE
ACTIVE ---- idle_seconds > scan_entry_idle_seconds ------------> SCAN
SCAN ---- blob detected ----------------------------------------> ACTIVE
ACTIVE/SCAN ---- FaultEventMsg ---------------------------------> SAFE
SAFE ---- ground command (Phase II) ---------------------------> IDLE
```

Mode transitions are applied in `ops/main.py` by draining `mode_queue` each iteration.

---

## Critical-Path Data Flow

The path from raw frame to gimbal command and stored artifact:

```
1. imaging_process
   +-- FlirBlackflyCamera.acquire_frame()    ->  RawFrameMsg
       +-- raw_frame_queue.put(msg)

2. _run_inference_process()  [ops/main.py -- same process as inference]
   +-- raw_frame_queue.get()                 ->  RawFrameMsg
   +-- apply_calibration(bands)              ->  calibrated bands (function call)
   +-- select_bands(calibrated)              ->  4-band tensor (function call)
   +-- compute_quality_flags(...)            ->  frozenset[FrameUsabilityTag]
   +-- ProcessedFrameMsg(...)                ->  in-memory object (no queue)
   +-- InferenceEngine.run(processed)        ->  Ok(InferenceResultMsg)
   +-- inference_queue.put(inference_result) ->  to controller_process
   +-- storage_queue.put(StorageWriteMsg)    ->  to storage_process

3. controller_process
   +-- inference_queue.get()                 ->  InferenceResultMsg
   +-- apply_safety_gates(blobs)             ->  filtered blobs
   +-- GimbalArbiter.step(state, result, now) -> (new_state, GimbalCommandMsg, events)
   +-- gimbal_queue.put(command)             ->  to hardware stub
   +-- telemetry_queue.put(events)           ->  to telemetry_process

4. storage_process  [concurrent with step 3]
   +-- storage_queue.get()                   ->  StorageWriteMsg
   +-- write raw_bands.npy + sha256 verify
   +-- write processed_tensor.pt + sha256 verify
   +-- manifest.append(JSON line)            ->  atomic commit point
   +-- downlink_queue.put(DownlinkItemMsg)   ->  to comms_process
```

---

## Design Decisions

### Preprocessing inside inference process

Preprocessing runs as a plain function call inside `_run_inference_process()` in `ops/main.py`.
Moving it to a separate process would require serializing `(C, H, W)` float32 numpy arrays
through `multiprocessing.Queue` (pickle round-trip) on every frame -- unacceptable on the
hot path. The function-call approach has zero serialization cost.

### Pure-function gimbal arbiter

`GimbalArbiter.step()` is a pure function with no side effects. This makes the arbiter
deterministic, trivially unit-testable (no mocks needed), and replay-able from logs. All
mutable state lives in `ArbiterState`, which is passed in and returned. The caller
(`process.py`) owns the queue, the clock, and the state variable.

### Greedy IoU blob matching

Blob matching uses greedy intersection-over-union (descending sort, first match wins) rather
than optimal Hungarian assignment. For the expected case of < 10 blobs per frame, greedy
matching is O(n^2) with negligible cost and is fully deterministic. Hungarian assignment
would add a scipy dependency and complexity for no practical benefit at this blob count.

### EMA asymmetric initialization

The EMA centroid filter returns the raw observation unmodified on the first frame
(`initialized = False` -> return new_centroid). This avoids a "phantom history" effect where
the first smoothed output is biased toward an arbitrary initial state. Behavior differs
between frame 1 and all subsequent frames.

### Comm window as weekday gate

The TDRSS comm window is enforced as a UTC weekday check (`MON-FRI`). This is not an
approximation -- ISS data dumps are constrained to weekdays by the ISS-ground interface
protocol, and the dump schedule is fixed. No orbital contact prediction is needed or correct.

### Shared downlink queue

Both storage and telemetry write to the same `downlink_queue` (a `queue.PriorityQueue`).
The queue is thread-safe; no synchronization is needed. A single queue keeps priority
ordering global -- telemetry health packets at priority 0 always drain before imagery at
priority 2 or 3.

### Config loaded once at startup

`ops/config_loader.py` loads and merges `config/default.toml` + `config/flight.toml` once
at startup, producing frozen `PactConfig` dataclass instances distributed to each subsystem
as constructor arguments. No subsystem reads TOML at runtime. No dynamic reconfiguration --
changes require restart. This guarantees all processes see consistent configuration.

### Staged model deployment with rollback

Model updates follow a three-path layout: `active.pt` (running model), `staged.pt`
(uploaded but not yet activated), `rollback.pt` (last known good). Activation requires
SHA-256 integrity check + smoke test. On smoke test failure the system stays on `active.pt`
and emits `MODEL_CORRUPT`. On any post-activation fault, `rollback.pt` is restored
automatically. This directly mitigates RISK-COMM-004 and satisfies REQ-AIML-HIGH-004/005.

### Persistence gates as chatter suppression

The `acquire_persistence_frames = 3` and `release_persistence_frames = 5` thresholds are
the primary software mitigation for RISK-AIML-003 (detection oscillation → gimbal chatter).
A blob must appear in 3 consecutive frames before tracking begins, and disappear for 5
consecutive frames before tracking releases. These values are tunable in `config/default.toml`
and must be re-validated against the trained model's false-positive rate before flight.

### Graceful degradation on power loss

ISS can remove power from the PC at any time without warning (RISK-ELEC-002). The storage
subsystem uses an append-only manifest with atomic commit: a frame is only considered
stored once the manifest line is written and fsync'd. On restart after unexpected power
loss, the manifest is the ground truth — any frame not in the manifest is treated as
incomplete and discarded. This prevents silent data corruption without requiring a
transactional filesystem.

---

## Configuration Reference

Key non-obvious parameters. Full defaults in `config/default.toml`.

| Section | Key | Default | Effect |
|---------|-----|---------|--------|
| `[controller]` | `confidence_gate` | 0.55 | Min mean blob confidence (post-sigmoid). Blobs below this are filtered before the arbiter. |
| `[controller]` | `ema_alpha` | 0.4 | EMA smoothing: 0.4 × new + 0.6 × prev. Higher = more responsive, less filtering. |
| `[controller]` | `min_deadband_px` | 20 | Displacements < 20 px produce no gimbal command (jitter suppression). |
| `[controller]` | `max_deadband_px` | 250 | Displacements > 250 px trigger `GIMBAL_RUNAWAY` fault. |
| `[controller]` | `retarget_rate_limit_hz` | 0.5 | Max gimbal command rate (1 command / 2 s). Prevents motor overload. |
| `[controller]` | `acquire_persistence_frames` | 3 | Consecutive frames with blob before ACQUIRING → TRACKING. |
| `[controller]` | `release_persistence_frames` | 5 | Consecutive misses before TRACKING → IDLE. |
| `[controller]` | `scan_entry_idle_seconds` | 60.0 | Seconds in IDLE before transitioning to SCAN state. |
| `[controller]` | `scan_slew_rate_deg_per_s` | 0.5 | Slew rate during nadir scan mode (REQ-CTRL-SLEW-002). Hard maximum. |
| `[controller]` | `max_slew_rate_deg_per_s` | 2.0 | Max slew rate during retarget maneuvers (REQ-CTRL-SLEW-001 placeholder — update after hardware selection). |
| `[controller]` | `max_slew_deg_s` | 2.0 | Alias used by LQR output saturation; must match `max_slew_rate_deg_per_s`. |
| `[controller]` | `blob_iou_match_threshold` | 0.25 | Min IoU for greedy blob-to-track association. Below this, blob is treated as new. |
| `[controller]` | `min_blob_area_px` | 15 | Minimum blob pixel area to pass the area safety gate. |
| `[controller]` | `kalman_dt_s` | 0.1 | Kalman filter sample period. Must equal control loop period. **(TBD after hardware selection.)** |
| `[controller]` | `kalman_process_noise` | 0.01 | Q matrix diagonal — process noise covariance. Tune to observed gimbal jitter. |
| `[controller]` | `kalman_measurement_noise` | 0.1 | R matrix diagonal — encoder measurement noise covariance. Tune to encoder noise floor. |
| `[controller]` | `lqr_Q_diag` | [10, 10, 1, 1] | LQR state cost weights [θ_az, θ̇_az, θ_el, θ̇_el]. Higher position weights (10) penalize pointing error heavily. |
| `[controller]` | `lqr_R_diag` | [1.0, 1.0] | LQR control effort weights [u_az, u_el]. Increase to reduce motor activity. |
| `[inference]` | `latency_budget_ms` | 500.0 | **Placeholder** — pending Jetson Xavier benchmark. `InferenceEngine.run()` returns `Err(INFERENCE_TIMEOUT)` if exceeded. |
| `[fault]` | `watchdog_interval_s` | 5.0 | Heartbeat send interval per subsystem. |
| `[fault]` | `watchdog_max_miss_count` | 3 | Missed heartbeats before `PROCESS_DIED` fault (15 s total). |
| `[fault]` | `thermal_limit_c` | 80.0 | Xavier thermal limit; triggers `check_thermal()` fault. Separate from gimbal motor thermal limits (TBD). |
| `[fault]` | `power_limit_w` | 55.0 | System power limit; triggers `check_power()` fault. Headroom below 60 W hard cap (REQ-RESO-HIGH-001). |
| `[comms]` | `max_daily_downlink_bytes` | 1,073,741,824 | 1 GB/day cap enforced by `DownlinkQueue.dequeue()`. |
| `[comms]` | `comm_window_days` | MON–FRI | Weekday-only downlink gate per ISS interface protocol (REQ-COMM-HIGH-002). |

---

## Test Strategy

Three tiers, all in `tests/`:

**Unit tests** -- isolated, pure-function tests with no real processes. MockCamera replaces
`FlirBlackflyCamera`. All threshold-sensitive functions are parameterized with below/at/above
boundary values. Result types are unwrapped with `assert isinstance(result, Ok)` before
accessing `.value`. Fixture catalogue in `tests/conftest.py`.

**Integration tests** -- multi-subsystem pipeline tests:
- `test_inference_pipeline`: full preprocessing + inference chain with a synthetic frame
- `test_controller_pipeline`: `InferenceResultMsg` -> safety gates -> arbiter -> command
- `test_comms_pipeline`: downlink queue priority ordering + CCSDS packet round-trip

**E2E smoke test** (`tests/e2e/test_full_pipeline_smoke.py`) -- all processes spawned
simultaneously. Synthetic `InferenceResultMsg` is injected onto the inference queue for the
first 3 frames to force ACQUIRING/TRACKING transitions (the untrained model produces random
output). Marked `@pytest.mark.e2e`, 60-second timeout.

### Controller-Specific Test Requirements

The following controller tests are required but not yet written (tracked in `TODO.md`):

| Test | What to Verify |
|------|---------------|
| EMA asymmetric initialization | Frame-1 output equals raw centroid; frame-2 output differs for same input. Documents intentional behavior. |
| Threshold boundary parameterization | All confidence gate, deadband, and rate-limit tests must be parameterized with below/at/above boundary values — not just happy path. |
| `DownlinkPriority` queue ordering | Enqueue items at all four priority levels out of order; assert dequeue order is always `HEALTH_TELEMETRY` first. |
| Storage date-rollover edge case | Inject `StorageWriteMsg` with timestamp straddling midnight; assert two separate date subdirectories are created. |
| LQR DARE fallback | Assert `LqrController.from_config()` returns `Err(CONTROLLER_FAULT)` when DARE solver fails, not a silent proportional fallback. |
| Encoder loss fault | Simulate missing encoder feedback for **(TBD)** cycles; assert controller halts actuator commands and arbiter transitions to SAFE. |
| Control loop watchdog | Simulate stalled loop; assert watchdog fires within **(TBD)** ms and emits safe halt. |
| Hard/soft limit enforcement | Command beyond hard limit; assert output is clipped and `limit_reached` flag is set in telemetry. |

---

## Known Gaps / Phase II

| Subsystem    | What's Stubbed                                   | Impact |
|--------------|--------------------------------------------------|--------|
| imaging      | `FlirBlackflyCamera` PySpin integration          | All CI runs use `MockCamera`; no hardware path exercised |
| model        | TensorRT INT8 quantization (`quantize.py`)       | Inference runs in FP32; Jetson latency uncharacterized |
| model        | `latency_budget_ms` (500 ms placeholder)         | Timeout fault threshold is unvalidated |
| preprocessing| `MOTION_SMEAR` detection                         | Always absent from quality flags |
| comms        | TDRSS modem hardware interface                   | Downlink writes to file/socket only |
| comms        | CCSDS secondary headers + CRC-16/CCITT           | Primary header only; payload uses `pickle.dumps` |
| comms        | Uplink chunk reassembly timeout                  | Incomplete uplinks accumulate indefinitely |
| storage      | Compression for raw `.npy` files                 | High disk consumption per frame |
| storage      | LRU eviction on storage full                     | `STORAGE_FULL` fault halts all writes permanently |
| telemetry    | Thermal/power sensor HAL                         | Both fields are hardcoded `0.0` |
| fault        | Safe-mode exit via ground command                | System cannot exit safe mode autonomously |
| ops          | Process restart on crash                         | Crashed subsystem transitions system to safe mode; no recovery |
| controller   | `send_gimbal_command()` hardware driver          | Stub only; blocked on gimbal hardware selection |
| controller   | LQR plant model (inertia J)                      | TBD; requires gimbal hardware characterization |
| controller   | All REQ-CTRL-POINT/SLEW/LIMIT/FAULT thresholds   | All numeric thresholds are TBD pending gimbal selection |
| controller   | Control loop rate                                | TBD; must be ≥ 2× gimbal mechanical bandwidth |
| controller   | `LqrController.from_config()` DARE fallback bug  | Silently falls back to proportional gain on solver failure; should return `Err(CONTROLLER_FAULT)` — see TODO.md |
| controller   | SAFE state in `GimbalArbiter`                    | Not yet implemented; required by REQ-CTRL-FAULT-001 |
| controller   | Motor temperature monitoring                     | REQ-CTRL-FAULT-003 requires motor temp polling; no HAL yet |

---

## Planned Subsystems (Phase II)

### Power / Thermal Management

The Jetson Xavier reports real-time power draw via the INA3221 sensor on the carrier board.
Power consumption is a reliable proxy for thermal load. A future `power/` subsystem would:
- Poll Xavier power draw periodically via the system power sensor interface
- Feed real watt readings into `fault/detector.py` `check_power()` and `check_thermal()`
  (currently called with mocked `0.0` values)
- Replace the placeholder thermal limits in `fault/` with real thresholds derived from
  Xavier thermal envelope specifications

This removes the last two mocked fault detectors and completes the fault detection loop.
