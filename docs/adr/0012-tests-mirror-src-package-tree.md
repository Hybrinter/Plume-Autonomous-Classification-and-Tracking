# ADR-REPO-0012: Tests mirror the src package tree

**Status:** Accepted
**Date:** 2026-08-25
**Topic:** restructure
**Supersedes:** none
**Superseded-by:** none
**Related:** ADR-REPO-0011

## Context

Each workspace member uses src layout. Hatch packs only `src/<package>` into the wheel.
The test suite lived as a flat list under `packages/<member>/tests/`. That layout hid the
mapping from a source module to its tests as the suite grew.

Sibling tests next to production modules would sit inside the installable package. They
would enter the `pact-flight` wheel. They would also join the `flight` import graph that
import-linter and mypy already treat as production code.

## Decision

Place tests **outside** `src/`. Nest them so they mirror the path **inside** `src/<package>/`.
Omit the installable package name as a tests folder:

```
packages/flight/src/flight/payload/gimbal/arbiter.py
packages/flight/tests/payload/gimbal/test_arbiter.py
```

Do not use `packages/flight/tests/flight/...`. A second `flight/` tree under tests would
compete with `packages/flight/src/flight/` for the `flight` module.

Keep original `test_*.py` basenames. Do not add `__init__.py` under `tests/`. Keep pytest
`--import-mode=importlib`. Keep `conftest.py` at the member `tests/` root when fixtures are
shared.

Tests that are not twins of a source module stay at that member's `tests/` root. Examples:
script checks, import-linter guards, package smoke imports, and GSE scenario-file runs.

GSE source is already flat under `src/gse/`. Its tests stay flat under `packages/gse/tests/`.

## Consequences

- A reader can walk from a source file to its tests by the same relative path under `tests/`.
- The flight wheel still contains only `src/flight`.
- VCRM evidence entries use the nested repo-relative paths.
- Pytest `testpaths` stay `packages/*/tests`; collection is recursive.

## Alternatives considered

- **Sibling tests inside `src/`** — ships tests on the flight image and folds them into the
  `flight` package for mypy and import-linter.
- **`tests/<package>/...` as a true src mirror** — a second `flight` (or `sim`) tree under
  tests collides with the source package under mypy namespace packages.
- **Keep a flat `tests/` dump** — fewer directories, but the mapping from module to test is
  only in the filename.
