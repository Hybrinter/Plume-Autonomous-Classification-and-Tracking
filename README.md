# PACT — Plume Autonomous Capture Technology

> Software architecture, requirements, risk mitigations, and implementation spec for the PACT onboard ISS plume detection and gimbal control system.

PACT is an active external payload hosted on the **TAMU-SPIRIT Pallet Carrier** aboard the International Space Station. It autonomously detects and tracks industrial smoke plumes in multispectral imagery using an onboard neural network, drives a gimbal to keep plumes in frame, stores imagery with integrity checksums, and downlinks data to the ground for ML retraining — all with no real-time ground-in-the-loop control.

---

## What It Does

| Capability | Description |
|-----------|-------------|
| **Plume Detection** | U-Net/ResNet-34 segmentation model classifies smoke, wildfire, natural gas, and industrial stack plumes in 4-band VNIR imagery (490–842 nm) |
| **Gimbal Control** | LQR + Kalman filter closed-loop controller keeps detected plumes centered in the camera FOV |
| **Onboard Storage** | Frames stored with SHA-256 checksums and append-only manifests for dataset integrity |
| **CCSDS Downlink** | Priority-queued data downlink via TAMU-SPIRIT/ISS link (≤ 1 GB/weekday) |
| **Model Uplink** | Safe model update pipeline with staged deployment, integrity verification, and rollback |
| **Fault Detection** | Heartbeat watchdog, encoder loss detection, thermal/power monitoring, safe-mode entry |

---

## Repository Structure

```
pact/
├── config/
│   ├── default.toml          # All tunable parameters — no magic numbers in source
│   └── flight.toml           # Flight overrides (INT8 quantization enabled)
├── src/pact/
│   ├── controller/           # Gimbal arbiter, LQR, Kalman filter, blob tracker
│   ├── imaging/              # FLIR Blackfly S GigE camera interface
│   ├── model/                # U-Net segmentation model, inference engine
│   ├── preprocessing/        # Band selection, radiometric calibration, quality flags
│   ├── comms/                # CCSDS encoding, downlink queue, chunked uplink
│   ├── storage/              # Frame persistence with SHA-256 and manifests
│   ├── telemetry/            # Health event aggregation and telemetry packets
│   ├── fault/                # Watchdog, fault detection, safe-mode FSM
│   ├── ops/                  # Process orchestrator, config loader, mode FSM
│   └── types/                # Shared enums, frozen message dataclasses, config types
├── tests/                    # Unit, integration, and E2E smoke tests
├── scripts/                  # Demo scripts, dataset download, benchmarks
├── docs/
│   └── architecture.md       # Full software architecture and implementation spec ← start here
├── CLAUDE.md                 # Codebase-wide patterns for contributors
└── TODO.md                   # Known gaps and Phase I/II work items
```

---

## Getting Started

### Install

```bash
# Requires Python 3.14+
uv sync --extra dev
```

### Run the Demos

```bash
# Gimbal arbiter state machine (no hardware needed)
python scripts/demo_controller.py

# Single inference pass on a synthetic frame
python scripts/demo_inference.py

# Storage write + SHA-256 verify
python scripts/demo_storage.py

# CCSDS packet encode/decode + CRC-32 verify
python scripts/demo_ccsds.py
```

### Run Tests

```bash
# Fast unit + integration tests
pytest -m "not e2e"

# Full pipeline smoke test (all processes, 60s timeout)
pytest -m e2e
```

---

## Key Design Principles

**Rust-idiomatic Python.** Every design choice optimizes for mechanical translation to Rust: frozen dataclasses, enum discriminants, `Result[T, E]` error handling, no dynamic dispatch.

**No magic numbers.** Every threshold lives in `config/default.toml`. Source code never hardcodes a tunable value.

**Pure-function arbiter.** `GimbalArbiter.step()` is a pure function — no I/O, no side effects. Deterministic, testable, and replayable from logs.

**Queue ownership in one place.** All inter-process queues are created in `ops/main.py`. No subsystem creates its own queues.

---

## Platform

| Parameter | Value |
|-----------|-------|
| Host platform | TAMU-SPIRIT Pallet Carrier, ISS external truss |
| Compute | NVIDIA Jetson Xavier NX |
| Camera | FLIR Blackfly S GigE 5MP (Sony IMX264, 3.45 µm pixel pitch) |
| Spectral bands | 490 / 560 / 665 / 842 nm (2×2 Torrent Photonics filter array) |
| Orbital altitude | ~420 km |
| Ground sampling distance | ≤ 10 m |
| Power budget | ≤ 60 W (hard), ≤ 15–20 W (target) |
| Downlink | ≤ 1 GB/weekday via TAMU-SPIRIT/ISS TDRSS link |

---

## Documentation

- **[`docs/architecture.md`](docs/architecture.md)** — Full software architecture, subsystem design, requirements traceability, risk mitigations, and implementation spec. Read this first.
- **[`CLAUDE.md`](CLAUDE.md)** — Cross-cutting patterns that apply project-wide (queue ownership, pure-function contract, Result usage, config distribution).
- **[`TODO.md`](TODO.md)** — Phase I bugs, test coverage gaps, and Phase II work items.

---

## Team

| Area | Owner |
|------|-------|
| Gimbal Control | Vin Manoj Nair |
| Imaging | Param Patel |
| AI/ML | Aiden Kampwerth |
| Communications | Vin Manoj Nair |
| Power / Electrical | Matthew Thanjan |
| Structures | Riley McNew |
| Thermal / Safety | Steven Minniear |
