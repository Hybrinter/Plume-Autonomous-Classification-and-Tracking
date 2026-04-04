# ADR-001: Centralized Config Load at Startup

**Status:** Accepted
**Date:** 2026-04-03
**Req IDs:** REQ-OPER-HIGH-002 (implied), general coding standard §3.6

## Context

PACT has approximately 20 tunable parameters across 5 subsystem config dataclasses
(`ControllerConfig`, `InferenceConfig`, `CommsConfig`, `StorageConfig`, `FaultConfig`).
These must be loaded from TOML, validated, and made available to each subsystem at process
startup.

Three config distribution patterns were considered:
1. **Each process loads its own TOML** — simple, but every process reads disk at startup,
   config validation logic is duplicated, and a bad config may not be caught until a
   specific process starts.
2. **Centralized load + distribution as typed dataclasses** — `ops/main.py` loads and
   validates config once, then passes each process only its relevant config slice as a
   typed dataclass argument.
3. **Environment variables / runtime injection** — too dynamic; violates the "no magic
   numbers" and "no dynamic dispatch" coding conventions.

Additionally, `config/flight.toml` must be merged on top of `config/default.toml` for
flight operation (e.g., enabling INT8 quantization). The merge logic must live in one place.

## Decision

`ops/config_loader.py` is the single point of config loading:
- Loads `config/default.toml` unconditionally.
- Merges an optional override TOML (e.g., `config/flight.toml`) on top.
- Validates all fields and returns `Result[PactConfig, str]`. Returns `Err` if any field
  is missing or out of valid range; `main()` calls `.unwrap()` to crash-fast on bad config.
- Distributes config to each process as a typed dataclass argument to the process entry point.
- No subsystem reads TOML directly. No subsystem has access to config fields outside its
  own config dataclass.

## Consequences

### Positive
- Config validation fails fast at startup before any process is spawned, preventing partial
  startup with a bad config.
- Each process entry point has a typed, explicit config argument — no global config state,
  no config singletons, no `os.environ` lookups.
- Flight vs. development config is handled by a single merge operation in one function.
  No conditional logic scattered across subsystems.
- The `PactConfig` dataclass hierarchy maps directly to the TOML section hierarchy
  (`[controller]`, `[inference]`, etc.), making the loader straightforward to implement
  with `tomllib.load()` and explicit field mapping.

### Negative / Trade-offs
- If a subsystem needs a new config field, `config.py`, `config_loader.py`, `default.toml`,
  and `flight.toml` must all be updated. Four files instead of one. This is intentional —
  the explicitness is the point.
- `config_loader.py` must explicitly map every TOML key to a dataclass field. Autogeneration
  (e.g., via `dacite` or `pydantic`) was considered but rejected: it adds a dependency and
  hides the mapping, making it harder to catch key name mismatches at review time.
- Config is immutable after startup. Dynamic config updates (e.g., changing a threshold via
  uplink command) are out of scope for Phase I and would require a new ADR.
