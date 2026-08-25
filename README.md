# PACT — Plume Autonomous Capture Technology

PACT is an **ISS-attached external payload** for autonomous detection, segmentation, and tracking
of industrial plumes in multispectral VNIR imagery from orbit. It runs a neural detector on each
frame, drives a two-axis gimbal to keep detected plumes boresighted, persists science products
with integrity checksums, and exchanges authenticated commands and downlink products with the
station over a CCSDS link — with no real-time ground-in-the-loop control.

The flight software is a **Python-only `uv` workspace** under `packages/`, built as isolated
**subsystem-apps** communicating over a typed in-process message bus. (The pre-restructure
`src/pact` architecture and the earlier Rust-migration plan have both been retired.)

---

## What It Does

| Capability | Description |
|-----------|-------------|
| **Plume detection** | An ONNX segmentation model (trained in a separate model repository) produces a binary plume mask per frame; blobs are extracted for tracking. Raw 2×2 mosaic frames are demosaiced into BLUE/GREEN/RED/NIR bands (≈ Sentinel-2 B2/B3/B4/B8) in pure preprocessing. |
| **Closed-loop pointing** | Boresight-relative error → EMA / Kalman tracking → LQR rate commands drive the gimbal; a pure FSM arbiter resolves IDLE / ACQUIRING / TRACKING / SCAN / SAFE behind safety gates. |
| **ISS command path** | Authenticated CCSDS command ingress (CRC + per-source sequence dedup + HMAC-SHA256 + command-dictionary validation); every command yields an ACCEPTED or REJECTED ack. |
| **FDIR / SAFE** | Heartbeat watchdog + fault-to-mode policy; SAFE-triggering faults latch the system into a single SAFE mode (stow + quiesce), exited only by ground command. |
| **Validation** | A configuration matrix of driver / compute profiles with a requirement → venue VCRM. A deterministic in-process SIL and a `sil-link-real` x86 partial run in CI, driven by declarative scenarios through the GSE harness. |

---

## Repository Structure

A `uv` workspace; the repo root is a **virtual** workspace root (builds no package):

```
packages/
  flight/   # pact-flight — the flight software (subsystem-apps; lean deps: numpy, scipy, structlog)
  sim/      # pact-sim    — SIL harness, scene generation, validation harness (depends on flight)
  tools/    # pact-tools  — artifact acceptance / SIL experiment runners / analysis
  gse/      # pact-gse    — ground support: CCSDS station emulator + declarative scenarios + orchestrator
config/     # default.toml (+ flight.toml override) — all tunable parameters, no magic numbers in source
profiles/   # sil / sil-link-real (run) + pil / hil (defined, not run) environment profiles
scenarios/  # declarative validation scenarios (scene + command timeline + assertions)
scripts/    # check_vcrm.py, check_docs.py, check_adr.py, check_flight_image.py — CI gates
docs/       # package-mirrored descriptive docs, ADRs, requirements (VCRM), validation
```

Each subsystem under `packages/flight/src/flight/` (`payload`, `fault`, `iss_iface`, `thermal`,
`electrical`, `mechanical`) is an isolated app: a thin imperative shell around a pure decision
core, talking to peers **only** over `flight.libs.bus`. `flight.core` is the sole composition root.

---

## Getting Started

Requires Python 3.14+ and `uv`.

### Full workspace (laptop / CI)

```bash
uv sync --extra dev
```

Run the gates (the whole tree):

```bash
uv run ruff check packages scripts
uv run ruff format --check packages scripts
uv run mypy packages scripts
uv run lint-imports
uv run python scripts/check_vcrm.py
uv run python scripts/check_docs.py --strict
uv run python scripts/check_adr.py --strict
uv run python scripts/check_flight_image.py
uv run pytest -m "not e2e"
```

### Flight-only (payload computer / experiment image)

```bash
uv pip install "./packages/flight"
```

Optional extras `inference`, `camera`, and `gimbal` exist as empty roles. Fill them when an onboard
runtime is chosen:

```bash
# later, when the onboard runtime is chosen:
# uv pip install "./packages/flight[inference]"
```

Workspace equivalent:

```bash
uv sync --package pact-flight --no-dev
```

### Training box

```bash
uv sync --package pact-tools --extra train
```

The `train` extra is empty until a training stack is chosen.

Run a declarative SIL scenario through the GSE harness (from the repo root):

```python
from gse.scenario import load_scenario
from gse.orchestrator import run_scenario

scenario = load_scenario("scenarios/closed_loop_pointing.toml")
report = run_scenario(scenario, f"profiles/{scenario.profile}.toml")
print(report.passed, report.failed, report.skipped)
```

---

## Architecture Principles

- **Subsystem-app + typed bus.** No app references another; all inter-app communication is a typed
  message on the `MessageBus`. Composition (bus, clock, drivers, scheduler) lives only in
  `flight.core` (and `sim.sil` for SIL).
- **Pure decision cores.** Controller, arbiter, tracking, watchdog, and policy are pure functions —
  no I/O, no bus, no clock reads; state in, state + messages out — so they are deterministic and
  replayable from logs.
- **HAL Protocols + lazy SDKs.** Every device is a `@runtime_checkable` Protocol returning
  `Result[..., FaultCode]`; real vs sim drivers are injected by the composition root through one
  `select_drivers(config)` factory. SDK imports (PySpin, onnxruntime, pyserial) are lazy.
- **Result, not exceptions.** Library code returns `Result[T, E]`; only process entry points raise,
  and only for unrecoverable startup failures.
- **Validation as a configuration matrix.** Profiles are points in a {sensor, gimbal, compute,
  link, clock} space; a requirement → method → venue VCRM (`docs/requirements/vcrm.md`) is the
  organizing spine, enforced in CI by `scripts/check_vcrm.py`.

Reference hardware (FLIR Blackfly S mono + 2x2 mosaic filter, PTU-class gimbal, Jetson Orin
NX-class compute, Ethernet/CCSDS station link) is nominated in package and validation docs.

---

## Documentation

- **[`docs/README.md`](docs/README.md)** -- documentation map (start here).
- **[`docs/flight.md`](docs/flight.md)** / [`sim`](docs/sim.md) / [`gse`](docs/gse.md) /
  [`tools`](docs/tools.md) -- package-mirrored as-is descriptions.
- **[`docs/style/ste-guide.md`](docs/style/ste-guide.md)** -- STE100-inspired writing rules.
- **[`docs/requirements/vcrm.md`](docs/requirements/vcrm.md)** -- requirement to venue matrix.
- **[`CLAUDE.md`](CLAUDE.md)** -- cross-cutting patterns and invariants for contributors.

Design decisions are recorded in the repository and package decision indexes linked from
[`docs/README.md`](docs/README.md). Descriptive pages do not link to those records.

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
