# ADR-REPO-0013: Pydantic dataclasses for TOML file schemas

**Status:** Accepted
**Date:** 2026-08-28
**Topic:** dependency
**Supersedes:** none
**Superseded-by:** none
**Related:** ADR-REPO-0010, ADR-REPO-0011, ADR-TOOLS-0002

## Context

Flight, GSE, and tools parse TOML with stdlib `tomllib` and hand-written mapping into frozen
dataclasses. Flight `load_config` re-lists every field in the dataclass tree, in range checks,
and in `_build_pact_config`. GSE scenarios and tools manifests repeat the same pattern at
smaller scale. A new field requires edits in several places, and a typo'd key is easy to miss
until a dedicated unknown-key check runs.

The payload computer is an NVIDIA Orin. Flight stays a separate wheel (ADR-REPO-0011) and must
not pull training stacks. It does not need to be ultralean on third-party libraries that only
run at startup. GSE scenarios currently use stdlib only (ADR-REPO-0010). The dumped training
stack used pydantic-settings and YAML; tools rejected that stack (ADR-TOOLS-0002) in favor of
plain torch and frozen dataclass / TOML config.

## Decision

Use **Pydantic v2 dataclasses** as the schema layer for TOML ingest. Pin `pydantic>=2.12` on
`pact-flight` (Python 3.14 support). Sim, GSE, and tools pick it up through `pact-flight`.

- Keep `tomllib` as the parser. Deep-merge default and override tables in `load_config`, then
  `TypeAdapter(...).validate_python` on the merged dict.
- Mark file schemas `frozen=True` with `extra="forbid"`. Reject unknown sections and keys at
  load time. Coerce TOML integers into float fields (non-strict ingest).
- Flight `load_config` still returns `Result[PactConfig, str]`. Map `ValidationError` to `Err`.
  GSE and tools loaders still raise.
- Put range checks on `Field` constraints. Put cross-field rules (gimbal envelope, mosaic
  permutation, band membership) on model validators.
- Keep Python field defaults aligned with `config/default.toml`. The defaults-equality test
  remains the guard.
- Do **not** use pydantic-settings, environment-variable overlays, or YAML.
- Do **not** use Pydantic for bus messages, `Result`, arbiter / Kalman / control state, or HAL
  structs. Those stay stdlib frozen dataclasses.
- Do **not** model sweep-space files as `TrainConfig`. Expand list-valued axes first, then
  validate each trial. VCRM and `pyproject.toml` CI scripts stay stdlib.

## Consequences

- Adding a TOML field is a schema field plus a TOML default plus the defaults-equality test.
- The flight image gains `pydantic` and `pydantic-core`. Training frameworks stay out.
- `dataclasses.replace()` keeps working on config objects.
- Reviewers treat pydantic-settings or Pydantic on the bus as out of scope for this decision.

## Alternatives considered

- **Keep hand-written `tomllib` mapping** — no new flight dependency; the per-field mapper and
  validator lists keep growing.
- **A generic `dataclasses.fields()` constructor without Pydantic** — deletes the mapper, but
  range and cross-field checks stay as a parallel walker.
- **Pydantic `BaseModel`** — same validation, but tests and analysis code that call
  `dataclasses.replace()` would have to switch to `model_copy`.
- **pydantic-settings** — pulls environment-variable and secrets sources that this repo does
  not use; profiles are named TOML overlays.
