# PACT Software Architecture

## System Overview

PACT (Plume Autonomous Capture Technology) is an ISS external payload operating at ~420 km
orbital altitude. It autonomously detects industrial smoke-stack plumes in multispectral VNIR
imagery, drives an active gimbal to track detected plumes, stores imagery and metadata with
integrity checksums, and downlinks data for ground-based ML retraining — all with no real-time
ground-in-the-loop control.

**Compute hardware:** Nvidia Jetson Xavier
**Imaging hardware:** FLIR Blackfly S BFS-PGE-50S5M-C (GigE Vision)
**Comms link:** TDRSS at 5 Mbps down / 2 Mbps up, weekdays only
**Daily data budget:** 1 GB downlink / 100 MB uplink

---

## Subsystem Descriptions

| Subsystem        | Role                                                                 |
|------------------|----------------------------------------------------------------------|
| **types**        | Foundation: all enums, message dataclasses, config dataclasses       |
| **model**        | U-Net/ResNet-34 segmentation model — train, evaluate, run inference  |
| **preprocessing**| Band selection, radiometric calibration, quality flagging, ROI crop  |
| **controller**   | Gimbal arbiter state machine, blob tracker, EMA filter, safety gates |
| **imaging**      | FLIR camera interface, frame acquisition loop                        |
| **comms**        | CCSDS encoding, priority downlink queue, chunked uplink handler      |
| **storage**      | Frame persistence to disk with SHA-256 checksums and manifests       |
| **telemetry**    | Health aggregation and CCSDS telemetry packet formatting             |
| **fault**        | Heartbeat watchdog, fault detection, safe-mode entry/exit            |
| **ops**          | Top-level orchestrator: process spawning, mode management, config    |

---

## Process Topology

Preprocessing runs inside the inference process (same process, function call) to keep the
critical-path latency tight. All other subsystems run as separate OS processes or threads.

```
imaging_process        --[raw_frame_queue]-->      inference_process
                                                   (preprocessing runs here)
inference_process      --[inference_queue]-->      controller_process
controller_process     --[gimbal_queue]-->          (hardware gimbal driver stub)
controller_process     --[telemetry_queue]-->       telemetry_process
inference_process      --[storage_queue]-->         storage_process
storage_process        --[downlink_queue]-->        comms_process
telemetry_process      --[downlink_queue]-->        comms_process
comms_process          --[uplink_queue]-->          ops/main.py (model deployment)
any subsystem          --[fault_queue]-->           fault_process
any subsystem          --[heartbeat_queue]-->       fault_process
fault_process          --[mode_queue]-->            ops/main.py
```

Queue payloads are frozen dataclasses from `src/pact/types/messages.py`.

---

## Concurrency Model

| Subsystem     | Primitive                           | Rationale                                      |
|---------------|-------------------------------------|------------------------------------------------|
| imaging       | `threading.Thread` + `queue.Queue`  | I/O-bound camera reads; GIL not a bottleneck   |
| inference     | `multiprocessing.Process`           | GPU-bound; needs true process isolation (REQ-AIML-COMP-002) |
| controller    | `multiprocessing.Process`           | CPU-bound arbiter logic; isolated from GIL     |
| storage       | `threading.Thread` + `queue.Queue`  | I/O-bound disk writes                          |
| comms         | `asyncio`                           | Many concurrent I/O waiters; scheduler + queue |
| telemetry     | `threading.Thread` + `queue.Queue`  | I/O-bound aggregation and packet formatting    |
| fault         | `threading.Thread` + `queue.Queue`  | Timer-driven watchdog; low CPU usage           |

---

## Dependency Layer Order

```
types                          ← no internal imports (dependency root)
  ↓
model / preprocessing / imaging
  ↓
controller
  ↓
storage / telemetry / comms
  ↓
fault
  ↓
ops                            ← imports everything; top-level orchestrator
```

---

## Gimbal Arbiter State Machine

```
              blob detected, confidence gate passed,
              persistence < acquire_persistence_frames
IDLE ──────────────────────────────────────────────────> ACQUIRING
  ^                                                          |
  |   all blobs lost                persistence >=           |
  <──────────────────── TRACKING <── acquire_persistence ───┘
  |                        |
  |   no blobs for         |  blob detected
  |   release_persistence  |
  |   frames               v
  <─────────────────── (back to IDLE or continue TRACKING)

IDLE ──── idle > scan_entry_idle_seconds ──────────────> SCAN
SCAN ──── blob detected ────────────────────────────────> ACQUIRING

any state ──── fault signal ────────────────────────────> SAFE
SAFE ──── fault cleared ────────────────────────────────> IDLE
```

---

## Configuration

All runtime parameters live in `config/default.toml`. Flight overrides in `config/flight.toml`
are merged on top at startup. No subsystem reads TOML directly — `ops/config_loader.py` loads
the merged config once and distributes typed `PactConfig` dataclass instances.

---

## Subsystem CLAUDE.md Files

Each subsystem has a `CLAUDE.md` with purpose, requirement IDs, owned/consumed messages,
invariants, and known gaps.

- `src/pact/types/CLAUDE.md`
- `src/pact/model/CLAUDE.md`
- `src/pact/preprocessing/CLAUDE.md`
- `src/pact/controller/CLAUDE.md`
- `src/pact/imaging/CLAUDE.md`
- `src/pact/comms/CLAUDE.md`
- `src/pact/storage/CLAUDE.md`
- `src/pact/telemetry/CLAUDE.md`
- `src/pact/fault/CLAUDE.md`
- `src/pact/ops/CLAUDE.md`
