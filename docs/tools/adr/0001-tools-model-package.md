# ADR-TOOLS-0001: One tools.model package for train, export, and accept

**Status:** Accepted
**Date:** 2026-08-26
**Topic:** restructure
**Supersedes:** none
**Superseded-by:** none
**Related:** ADR-REPO-0004, ADR-REPO-0011

## Context

`pact-tools` already holds SIL telemetry analysis under `tools.analysis`. Model
training, ONNX export, and artifact acceptance belong in the same tools package
but must not mix with per-subsystem SIL plots. Empty `train` and `export` extras
exist as install roles with no modules.

## Decision

Place model training, export, acceptance, and model metrics in one package
`tools.model` under `packages/tools/src/tools/model/`. `tools.analysis` remains
SIL observability. `tools.accept` re-exports `tools.model.accept` for existing
imports. Install extras `train` and `export` attach to this package; they are
not separate directories.

## Consequences

- `python -m tools.model` is the CLI surface.
- Flight continues to import neither `tools` nor torch.
- Descriptive docs live under `docs/tools/model/`.

## Alternatives considered

- Sibling packages `tools.train` and `tools.export` — collides with the rule that
  all model work shares one module next to `tools.analysis`.
- A fifth uv workspace member `pact-train` — splits the freeze boundary and
  duplicates extras already named on `pact-tools`.
