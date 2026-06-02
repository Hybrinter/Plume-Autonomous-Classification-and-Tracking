# ADR 0002: Drop Bazel; `uv` workspace + import-linter

**Status:** Accepted (2026-05-30)

## Context

The reference inspiration (a satellite operator's C++ monorepo) uses Bazel for hermetic builds and
fine-grained visibility enforcement. PACT is a single-language Python payload maintained by a small
team. Bazel's hermeticity and cross-language strengths buy little here, while its setup and
maintenance cost is significant, and Python tooling for the same goals has matured.

## Decision

Use standard Python tooling: a **`uv` workspace** with three packages (`flight` lean, `sim`,
`tools` heavy) so the flight dependency set stays minimal (no torch/onnxruntime/matplotlib in the
flight image). Enforce package isolation and layering with **`import-linter`** contracts (the role
Bazel `visibility` would have played). `ruff` (lint + format), `mypy --strict`, and `pytest` are
the remaining gates, run in GitHub Actions scoped to `packages/`.

## Consequences

- `import-linter` contracts encode the architecture as testable rules: flight/sim/tools isolation,
  the `core > apps > hal.interfaces > libs` layer order, and "concrete drivers only from
  composition roots."
- `mypy_path` must point at the workspace `src` dirs so cross-package imports resolve to source
  (otherwise they fall back to `Any`; see the strong-typing rule).
- No hermetic/remote-cache build; acceptable for this scale.
