# ADR-REPO-0014: DriverConfig for HAL axes; environment reserved for simulated world

**Status:** Proposed
**Date:** 2026-09-11
**Topic:** rename
**Supersedes:** none
**Superseded-by:** none
**Related:** ADR-REPO-0010

## Context

ADR-REPO-0010 introduced a per-axis sim/real selector as `EnvironmentConfig` under a TOML
`[environment]` table and `PactConfig.environment`. That name collides with the planned
sim-only world-model layer (scene dynamics, ISS orbit context, sensor geometry) that is not
HAL driver wiring.

## Decision

- Rename the HAL axis selector to **`DriverConfig`**, exposed as **`PactConfig.drivers`** and
  the TOML table **`[drivers]`**.
- Keep field semantics unchanged: each axis (`sensor`, `gimbal`, `compute`, `link`, `clock`,
  `ephemeris`) remains `"sim"` or `"real"`; `host` stays provenance-only.
- Reserve **`environment`** for a future sim-only world-model config (not implemented in this
  change). No `packages/sim/src/sim/environment/` tree is added here.

## Consequences

- Profiles (`profiles/sil.toml`, `sil-link-real.toml`, `pil.toml`, `hil.toml`) use `[drivers]`.
- `select_drivers`, GSE, and SIL read `config.drivers`; descriptive docs and tests follow.
- ADR-REPO-0010 remains accepted with its historical `[environment]` / `EnvironmentConfig`
  wording; this record documents the rename without rewriting that body.

## Alternatives considered

- Keep `EnvironmentConfig` and add a separate world-model name — rejected: the HAL table would
  keep the overloaded term and confuse STE readers.
- Implement world models in the same PR — rejected: scope stays a rename only.
