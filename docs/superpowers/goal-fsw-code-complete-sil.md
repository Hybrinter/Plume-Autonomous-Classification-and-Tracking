# GOAL: Close all remaining code-only flight-software tasks and make the deterministic SIL exercise them end-to-end

REPO: PACT ISS-attached payload. Branch `fsw-restructure`. `packages/` (flight/sim/tools/gse)
is the entire codebase; Python 3.14; uv workspace.

## Read first (authoritative)
- Canonical target: `docs/superpowers/specs/2026-06-09-pact-flight-final-state-design.md`
- Decisions: `docs/adr/0001..0010`
- Current state + phase history: memory file `project_fsw_parity_effort.md`
- Invariants/rules: `CLAUDE.md` + `.claude/rules/`

## Treat as stale / wrong
- `src/pact` and repo-root `tests/` are DELETED; any `pact.*` reference is a bug.
- "SIL->PIL->HIL ladder" is superseded by the configuration-matrix model (spec Section 9 / ADR-0010).
- The 2026-06-06 baseline's "current" side predates the sensor-ingest, pointing, link/ingress,
  validation-matrix, and legacy-retirement phases (all since landed) -- do not trust it for status.
- Repo is Python 3.14: PEP 758 unparenthesized `except A, B:` is valid; do not "fix" it.

## Already built (do not redo)
Sensor ingest (Section 3); pointing/gimbal closed loop (Section 5); CCSDS link + authenticated
command ingress/ACK (Section 6 link+ingress); validation config-matrix + GSE + scenarios
(Section 9); legacy retirement (Section 10).

## Remaining code-only capabilities (the objective set, all from the spec)
- Section 6 command router in `flight.core`: route `CommandMsg` to the target app; ARM/EXECUTE
  two-step; inhibit re-check at actuation; `EXIT_SAFE` (SAFE is currently a one-way latch).
- Section 6 data system: core-hosted checksummed/quota'd storage + `StorageWriter`/`StorageReader`
  Protocols; prioritized AOS-gated downlink manager; reboot-surviving FDIR fault ledger.
- Section 6 model upload: chunked uplink reassembly (`iss_iface`) + stage/`ACTIVATE`/auto-rollback
  (`core`) + `ModelDeployState` telemetry.
- Section 4 model lifecycle: `tools/` artifact-acceptance gate (manifest + SHA-256 + I/O contract +
  golden-scene IoU + latency); `OnnxDetector` load-time hash/contract verify (`MODEL_CORRUPT`) +
  per-frame latency budget (`INFERENCE_TIMEOUT`).
- Section 5 mechanical: `LaunchLock` HAL Protocol + `SimLaunchLock` + mechanical app +
  bidirectional gimbal interlock; release as a hazardous ARM/EXECUTE command.
- Section 7 platform robustness: bounded bus queues + overflow policy; `schema_version` on
  envelopes; startup health-gate -> SAFE; scheduler thread supervision (restart-then-SAFE);
  SIGTERM ordered teardown.
- Section 7 config integrity: full `config_loader._validate()` -- ranges, cross-field,
  unknown-key rejection.

## Done when
Every capability above is implemented behind the existing architecture (subsystem-app + typed bus;
pure cores; HAL Protocols + lazy SDK; `Result[T,E]`; composition-root ownership; build_apps/Scheduler
stay driver-agnostic; gse imports only flight.libs+sim) AND exercised in the deterministic SIL /
declarative GSE scenarios (command routed->executed->acked; product stored->downlinked; SAFE
entered+exited; lock released; model uploaded->activated->rolled back). All whole-tree gates green:
`ruff check`, `ruff format --check`, `mypy`, `lint-imports`, `scripts/check_vcrm.py`,
`pytest -m "not e2e"` -- with new VCRM rows + scenarios for the new running-venue requirements.

## Out of scope (hardware-blocked / not code-only)
PIL/HIL execution; real-driver hardware validation; real calibration artifacts; `RealScalarSensor`
real driver; systemd/external supervisor; the full Section 8 requirements + hazard-analysis
documents (author only the VCRM rows the new SIL scenarios require).
