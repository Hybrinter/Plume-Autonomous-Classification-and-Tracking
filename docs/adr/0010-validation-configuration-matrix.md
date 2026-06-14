# ADR 0010: Validation as a configuration matrix with a VCRM spine

**Status:** Proposed (2026-06-13)

**Implements:** spec Section 9 (validation), reframed, of
`docs/superpowers/specs/2026-06-09-pact-flight-final-state-design.md`. Refines the spec's
originally-planned ADR-8 ("validation ladder").

## Context

The 2026-06-09 spec framed validation as a **SIL -> PIL -> HIL ladder** driven by "one shared
harness." Two facts make the literal-ladder framing the wrong shape:

1. **`build_apps` is already a per-device, driver-agnostic composition.** Its signature is
   `build_apps(config, bus, clock, drivers, monitored, calib, uplink_key)`, and the `Drivers`
   bundle injects one implementation per device (`sensor`, `gimbal`, `detector`, `station`,
   `thermal_sensor`, `power_sensor`). Nothing forces the six devices to be all-real or all-sim
   together. A validation configuration is therefore a *point in a multi-axis space*, and
   SIL/PIL/HIL are **corners** of that space -- not three monolithic rungs.

2. **No hardware exists, and one "real" axis needs none.** `RealStationLink` is a genuine
   TCP/UDP CCSDS transport built on the Python standard library (no SDK, no device). It can run
   headless on x86 in CI against a station emulator. `RealSensor` (PySpin) and `RealGimbal`
   (pyserial) require physical hardware and are HIL-only. So the matrix has at least one **x86
   partial** corner that runs today, between pure SIL and the unreachable hardware corners -- a
   structure a three-rung ladder cannot express.

The spec also predates the sensor-ingest, closed-loop-gimbal, and link-transport phases; the
real drivers it calls "stubs" are now built. `RealScalarSensor` is still a 0.0 stub and **no
`LaunchLock` driver exists** (Protocol, real, or sim).

## Decision

### 1. Validation is a configuration matrix; profiles are named corners

A validation run is a point in a six-axis space -- `{sensor, gimbal, compute (detector), link,
clock, lock}` -- each axis independently `sim` or `real`, plus a host-architecture attribute
(x86_64 vs Jetson aarch64) recorded as a deployment fact, not a code switch. SIL/PIL/HIL are
**named corners** (`profiles/*.toml`), each **named by the deviation it closes**:

| Profile | sensor | gimbal | compute | link | clock | host | Status |
|---|---|---|---|---|---|---|---|
| `sil` | sim | sim | sim | sim | manual | x86 | Runs every CI |
| `sil-link-real` | sim | sim | sim | **real** | manual | x86 | Runs in CI |
| `pil` | sim | sim | sim/real | real | real | Jetson | Defined, not run |
| `hil` | real | real | real | real | real | Jetson+bench | Defined, not run |

**Profiles do not nest.** The `lock` axis is **defined but inert** -- there is no launch-lock
device, so it is a permanent VCRM gap, not a selectable value. The "ladder" survives only as the
documented adoption order in which corners are brought online.

**Why:** the matrix is honest about what can and cannot run without hardware, lets a single
real-link corner (`sil-link-real`) be exercised in CI, and gives every requirement a precise
verification venue instead of a coarse rung.

### 2. `[environment]` + one `select_drivers` factory is the only wiring change

- A new `[environment]` config block (`EnvironmentConfig`, frozen/slots) carries one
  `"sim" | "real"` field per selectable axis (plus the inert `lock` and an informational `host`).
  Its Python defaults are **all-real** (the flight default) and mirror `config/default.toml`
  exactly, per the `test_config_defaults` invariant. Profiles are overrides that set only the
  axes they deviate on, merged by the existing `load_config(default, override)` path.
- `flight.core.select_drivers(config, clock, sim_inputs) -> Drivers` reads `config.environment`
  and, per axis, lazy-imports `flight.hal.drivers_real.*` **only for axes set "real"**, otherwise
  constructs the sim driver. It lives in the composition root, the one layer permitted to import
  concrete drivers (`drivers-from-composition-roots-only`).
- `sim_inputs` is a frozen bundle (rendered frames, housekeeping readings, detector/prob-mask,
  inbound packets) consumed only by "sim" axes. This reconciles "one factory" with the fact that
  sim drivers need scene-provided inputs config cannot carry; "real" axes ignore it.
- `flight.core.main` and the SIL/GSE in-process harness both obtain `Drivers` from
  `select_drivers`. **`build_apps`, the `Scheduler`, and every app are untouched** -- the matrix
  changes composition only.

**Why:** keeping selection in one composition-root factory preserves the invariant that apps never
know whether they got a real or sim driver, keeps SDK imports lazy (CI and the lean flight image
stay SDK-free), and adds no new code path to `build_apps`.

### 3. `packages/gse` is a new workspace package; `flight` and `sim` never import it

`gse` imports `flight.libs` and `sim`; **neither `flight` nor `sim` imports `gse`** (two new
import-linter `forbidden` contracts). It provides:

- **Station emulator** (`gse.station`): the station side of the exact protocol `RealStationLink`
  speaks -- authenticated CCSDS command uplink (HMAC/seq/CRC via `flight.libs.ccsds` + the command
  dictionary), telemetry/product capture, AOS/LOS scheduling. In-process over loopback sockets for
  `sil-link-real`; over real Ethernet from a bench PC for PIL/HIL (future).
- **Scenario format** (`gse.scenario`): declarative **TOML** -- a `[config]`/profile reference, a
  `[scene]` plume/kinematics script with seed, a `[[commands]]` timeline, and `[[assertions]]`
  expected outcomes.
- **Stepping seam** (`gse.harness`): a `HarnessBackend` Protocol (`build` / `step` /
  `inject_command` / `collect` / `shutdown`). The **`InProcessBackend` is implemented**
  (single-threaded `ManualClock` stepping in the style of the existing `SilHarness`, plus the
  in-process station emulator for the link-real corner). A **`SocketBackend` is defined but not
  implemented** -- the future PIL/HIL real-Ethernet backend -- so the seam exists without the
  plumbing. One scenario format, two backends, not one harness.
- **Orchestrator + analysis** (`gse.orchestrator`): runs a scenario against a profile, injects
  commands on the timeline, scores assertions, and emits a JSON V&V evidence record.

**Why TOML:** it matches the repo's existing config convention (`config/default.toml`,
`profiles/*.toml`), needs no new dependency (`tomllib` is stdlib), and keeps scenarios as portable
data artifacts the orchestrator scores rather than code.

### 4. Assertions are tagged for portability; only frame/event-counted ones port

Every scenario assertion is tagged **`frame-portable`** or **`realtime-only`**. Frame- and
event-counted assertions ("`TRACKING` within N frames", "`SAFE` latched", "ACK with valid CRC
observed") port across venues unchanged. Time-deadline and ordering assertions are `realtime-only`
and **re-authored per venue as bounds** -- SIL determinism is `ManualClock` + faked heartbeats, so
wall-clock deadlines are meaningless there. Under SIL the orchestrator records `realtime-only`
assertions as skipped-with-reason rather than passing them vacuously.

**Why:** a scenario "written once" must not silently claim a timing guarantee that the
deterministic in-process backend cannot evaluate.

### 5. The VCRM is the organizing artifact; a CI check enforces traceability

`docs/requirements/` carries a **VCRM**: `requirement -> statement -> verification method
(unit | SIL | PIL | HIL) -> venue/profile -> evidence (test or scenario id) -> status`. In this
effort it is populated **only for requirements the implemented `sil` and `sil-link-real` scenarios
exercise**; the complete requirements baseline (spec Section 8) stays future work. Module
docstrings cite REQ IDs and tests/scenarios tag them (existing convention), and a **CI check**
asserts that every VCRM requirement whose venue is a *running* profile is cited by at least one
module and one test/scenario, and that no requirement claims verification at a venue that does not
run. The permanent **"real ground segment never tested"** gap is a standing VCRM row.

**Why:** the VCRM, not the ladder, is what ties code and tests to verification venues; the CI
check keeps it from drifting into aspirational coverage.

## Scope of this ADR's implementation

**In:** `EnvironmentConfig` + `[environment]` + `select_drivers`; `profiles/{sil,sil-link-real}`
runnable in CI and `profiles/{pil,hil}` defined-not-run; `packages/gse` (station emulator + TOML
scenario + `InProcessBackend` + orchestrator/analysis); the `HarnessBackend` seam (in-process
backend only, `SocketBackend` stubbed); the thin VCRM slice + traceability CI check + the two
`flight`/`sim` -> `gse` import contracts.

**Out (deferred):** running PIL/HIL; the `SocketBackend` and Jetson/bench runners; the
`LaunchLock` driver; the complete requirements baseline; the model-acceptance harness; the full
data system; legacy `src/pact` retirement and CI widening. Any of these is pulled in only if a
validation exercise forces it, and surfaced when it does.

## Consequences

- **The matrix is honest about hardware.** Exactly two corners run (`sil`, `sil-link-real`); the
  rest are documented configurations, so CI never pretends to verify what no hardware can.

- **The real link is exercised in the closed loop, not just in isolation.** ADR-0009's loopback
  tests prove `RealStationLink` in unit isolation; `sil-link-real` proves it end-to-end through
  the flight apps and the GSE station emulator.

- **Composition stays the single wiring point.** `select_drivers` is additive; `build_apps`, the
  `Scheduler`, and the apps are unchanged, so the driver-agnostic invariant is preserved.

- **No new third-party dependencies.** `gse` and the scenario format use stdlib (`tomllib`,
  `socket`, `hmac`, `json`); `sil-link-real` needs no SDK or device.

- **The stepping seam is forward-compatible.** PIL/HIL adoption adds a `SocketBackend` behind the
  existing `HarnessBackend` Protocol; scenarios and frame-portable assertions carry over, and
  realtime-only assertions are re-authored as bounds at that point.

- **A standing gap is recorded, not hidden.** "Real ground segment never tested" is a permanent
  VCRM row, because the GSE stands in for the station at every corner.
