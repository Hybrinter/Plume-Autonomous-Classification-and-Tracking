# GOAL: PACT canonical FSW validation (config-matrix model)

Bring PACT (packages/flight) validation infra to a canonical state: a CONFIGURATION MATRIX of driver/compute profiles with a requirement->venue VCRM as the organizing spine. SIL/PIL/HIL are named profiles, NOT a literal ladder. No hardware exists: only SIL + x86 partial profiles RUN; PIL/HIL are DEFINED + documented, not run. Implement through SIL, then STOP for human review before PIL/HIL plumbing.

## Read first
- docs/superpowers/specs/2026-06-09-pact-flight-final-state-design.md (target; Section 9 is what you REFRAME; PREDATES ~4 done phases).
- docs/architecture.md + CLAUDE.md (invariants; non-negotiable).
ALREADY BUILT, don't re-derive: raw-mosaic ingest+demosaic (ADR-0007); RealSensor/RealGimbal+closed-loop pointing (ADR-0008)/RealStationLink (code-complete, hardware-unvalidated); iss_iface authenticated CCSDS command ingress (HMAC/seq/CRC/ACK-NACK); deterministic in-process SIL with CI closed-loop tests.

## Decision (Approach C)
build_apps(config,bus,clock,drivers,monitored) is ALREADY driver-agnostic + per-device (same in flight.core.main and sim.sil). Validation config is a point in {compute,sensor,gimbal,lock,link,clock}; SIL/PIL/HIL are corners; "ladder" survives only as documented adoption order. Build:
1. An [environment] config block + ONE select_drivers(config) factory lazy-importing drivers_real.* only for axes set "real". ONLY wiring change; build_apps/scheduler/apps untouched.
2. profiles/*.toml: sil + >=1 x86 partial profile runnable; pil + hil defined, not run.
3. packages/gse (new): station emulator (station side of RealStationLink's CCSDS) + DECLARATIVE scenario format (config/scene/command-timeline/assertions) + orchestrator/analysis. gse imports flight.libs+sim; flight NEVER imports gse (add import-linter contract).
4. VCRM (requirement->method->venue) as the organizing artifact.

## Guardrails (non-negotiable)
- Profiles DON'T nest; no profile tests the real ground segment (GSE stands in). Name each by the deviation it closes; log "real ground segment never tested" as a PERMANENT VCRM gap.
- Only frame/event-counted assertions port; TIME-deadline/ordering ones don't (SIL determinism = ManualClock + faked heartbeats). TAG each assertion frame-portable|realtime-only; re-author realtime ones per venue as bounds.
- One scenario format, TWO transport backends (not one harness). DESIGN the stepping seam now (shared SilHarness/future-PilHarness interface or steppable Scheduler.tick()); IMPLEMENT only the in-process backend.
- Honor all CLAUDE.md invariants; build_apps stays driver-agnostic; verify real signatures before relying on them.

## Scope
IN: select_drivers + [environment]/profiles; packages/gse; SIL + >=1 x86 profile in CI; seam interface (in-process only); VCRM + traceability conventions + CI check, populated for paths validation exercises; ADR + spec amendment.
OUT (defer; pull in only if validation needs it, and surface it): full data system, launch lock, legacy src/pact retirement + CI widening, model-acceptance harness, complete requirements baseline, running PIL/HIL, socket backend + Jetson/bench runners.

## Process (superpowers)
1. brainstorming -> amend spec Section 9 with C, reconcile delta, write ADR; PAUSE for human spec review.
2. writing-plans -> plan in-scope work.
3. TDD; run the CLAUDE.md gates (ruff/format/mypy/lint-imports/pytest -m "not e2e") after each unit.
4. verification-before-completion before any "done" claim.
5. CHECKPOINT: stop once SIL+gse green and PIL/HIL profiles+procedures documented; hand back before PIL/HIL implementation.

## Surface, don't guess
Profile axes/names + which partials to bless; scenario DSL (TOML/YAML/Python); requirements-baseline depth now vs later; any OUT item validation forces back IN.
