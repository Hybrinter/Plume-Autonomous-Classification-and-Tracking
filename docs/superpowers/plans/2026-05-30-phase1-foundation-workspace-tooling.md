# Phase 1 -- Foundation: uv Workspace & Tooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the single `pact` package into a uv workspace with lean `flight`, `sim`, and heavy `tools` members, lay out the empty `flight` subsystem-app skeleton, and wire every quality gate (ruff, mypy, pytest, import-linter) plus GitHub Actions CI -- all green -- without disturbing the existing `src/pact` code.

**Architecture:** The repo root becomes a uv **workspace** while remaining the transitional `pact` package (existing `src/pact` is untouched and migrated in later phases). Three new members live under `packages/`: `flight` (numpy + structlog only), `sim` (depends on flight), `tools` (depends on flight + sim). Each uses a `src/` layout. `import-linter` enforces the dependency spine: flight cannot import sim/tools, apps are mutually independent and sit above `hal.interfaces` above `libs`, and concrete drivers are reachable only from composition roots.

**Tech Stack:** Python 3.14, uv workspace, hatchling, ruff, mypy (strict), pytest, import-linter, GitHub Actions.

---

## Context for the implementer

- This repo already uses **uv** with a root `pyproject.toml` (build backend `hatchling`, the package is `pact` at `src/pact`). Do **not** create a new build system or delete `src/pact`.
- The existing root `pyproject.toml` already configures ruff (line-length 100, target `py314`, rules `E,F,I,N,UP,ANN`, ignoring `ANN101,ANN102`), mypy (`strict`, `python_version=3.14`), and pytest (`testpaths=["tests"]`, marker `e2e`). Reuse these; only extend them.
- Python is **3.14** and `requires-python = ">=3.14"`. Keep that for the new members.
- All new test functions must be annotated `-> None` to satisfy the `ANN` ruff rules already in force.
- The importable package names must be exactly `flight`, `sim`, `tools` (the `import-linter` contracts and the spec reference these module paths). Distribution names are `pact-flight`, `pact-sim`, `pact-tools`.

## File structure (created in this phase)

```
pyproject.toml                                  # MODIFY: add workspace table + import-linter dev dep
.importlinter                                   # CREATE: layering contracts
.github/workflows/ci.yml                        # CREATE: CI gates
packages/
  flight/
    pyproject.toml                              # CREATE
    src/flight/__init__.py                      # CREATE (empty)
    src/flight/libs/__init__.py                 # CREATE (empty)
    src/flight/libs/version.py                  # CREATE (first typed module)
    src/flight/hal/__init__.py                  # CREATE (empty)
    src/flight/hal/interfaces/__init__.py       # CREATE (empty)
    src/flight/hal/drivers_real/__init__.py     # CREATE (empty)
    src/flight/hal/drivers_sim/__init__.py      # CREATE (empty)
    src/flight/core/__init__.py                 # CREATE (empty)
    src/flight/payload/__init__.py              # CREATE (empty)
    src/flight/thermal/__init__.py              # CREATE (empty)
    src/flight/electrical/__init__.py           # CREATE (empty)
    src/flight/mechanical/__init__.py           # CREATE (empty)
    src/flight/iss_iface/__init__.py            # CREATE (empty)
    src/flight/fault/__init__.py                # CREATE (empty)
    tests/test_version.py                       # CREATE
  sim/
    pyproject.toml                              # CREATE
    src/sim/__init__.py                         # CREATE (empty)
    src/sim/sil/__init__.py                     # CREATE (empty)
    src/sim/scene/__init__.py                   # CREATE (empty)
    src/sim/twin/__init__.py                    # CREATE (empty)
    tests/test_import.py                        # CREATE
  tools/
    pyproject.toml                              # CREATE
    src/tools/__init__.py                       # CREATE (empty)
    tests/test_import.py                        # CREATE
```

---

## Task 1: Declare the uv workspace and add the import-linter dev dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the workspace table to the root `pyproject.toml`**

Add this block (anywhere after the `[project]` table; placing it next to the existing `[tool.uv]` block is tidy):

```toml
[tool.uv.workspace]
members = ["packages/flight", "packages/sim", "packages/tools"]
```

- [ ] **Step 2: Add `import-linter` to the existing `dev` extra**

In `[project.optional-dependencies]`, change the `dev` list to include import-linter (keep the existing entries exactly):

```toml
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.0",
    "pytest-timeout>=2.1",   # required for e2e test 60-second timeout assertion
    "mypy>=1.5",
    "ruff>=0.1",
    "import-linter>=2.0",
]
```

- [ ] **Step 3: Extend pytest test discovery to the new package test dirs**

In `[tool.pytest.ini_options]`, change `testpaths` to:

```toml
testpaths = [
    "tests",
    "packages/flight/tests",
    "packages/sim/tests",
    "packages/tools/tests",
]
```

- [ ] **Step 4: Verify the root file still parses**

Run: `uv run python -c "import tomllib, pathlib; tomllib.loads(pathlib.Path('pyproject.toml').read_text()); print('ok')"`
Expected: prints `ok` (no `TOMLDecodeError`). The workspace members do not exist yet, so do **not** run `uv sync` here -- it would fail. The sync happens in Task 5.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "build: declare uv workspace and add import-linter dev dep"
```

---

## Task 2: Create the `flight` member with its first typed module and test

**Files:**
- Create: `packages/flight/pyproject.toml`
- Create: `packages/flight/src/flight/__init__.py` and the empty subpackage tree (see Step 2)
- Create: `packages/flight/src/flight/libs/version.py`
- Test: `packages/flight/tests/test_version.py`

- [ ] **Step 1: Create `packages/flight/pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pact-flight"
version = "0.1.0"
description = "PACT flight software (lean flight image)"
requires-python = ">=3.14"
dependencies = [
    "numpy>=1.24",
    "structlog>=23.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/flight"]
```

- [ ] **Step 2: Create the empty package tree**

Create each of these files as an **empty** file (zero bytes is fine):

```
packages/flight/src/flight/__init__.py
packages/flight/src/flight/libs/__init__.py
packages/flight/src/flight/hal/__init__.py
packages/flight/src/flight/hal/interfaces/__init__.py
packages/flight/src/flight/hal/drivers_real/__init__.py
packages/flight/src/flight/hal/drivers_sim/__init__.py
packages/flight/src/flight/core/__init__.py
packages/flight/src/flight/payload/__init__.py
packages/flight/src/flight/thermal/__init__.py
packages/flight/src/flight/electrical/__init__.py
packages/flight/src/flight/mechanical/__init__.py
packages/flight/src/flight/iss_iface/__init__.py
packages/flight/src/flight/fault/__init__.py
```

- [ ] **Step 3: Write the failing test**

Create `packages/flight/tests/test_version.py`:

```python
"""Smoke test for the flight package version accessor."""

from flight.libs.version import flight_version


def test_flight_version_returns_semver() -> None:
    """flight_version returns the expected version string."""
    assert flight_version() == "0.1.0"
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `uv run pytest packages/flight/tests/test_version.py -v`
Expected: collection/import error -- `ModuleNotFoundError: No module named 'flight.libs.version'` (the module does not exist yet; `flight` is also not installed yet, which is resolved in Task 5 -- if the error is instead `No module named 'flight'`, that is also an acceptable failure for this step).

- [ ] **Step 5: Write the minimal implementation**

Create `packages/flight/src/flight/libs/version.py`:

```python
"""Flight package version accessor.

Provides the flight software version string. This is the first real flight
module and exists to exercise the quality gates against typed flight code.
"""

FLIGHT_VERSION: str = "0.1.0"


def flight_version() -> str:
    """Return the flight software version string.

    Returns:
        str: The semantic version of the flight package.
    """
    return FLIGHT_VERSION
```

- [ ] **Step 6: Commit (verification of the test runs after the workspace is synced in Task 5)**

```bash
git add packages/flight
git commit -m "feat(flight): scaffold flight member with version accessor"
```

---

## Task 3: Create the `sim` and `tools` members with import smoke tests

**Files:**
- Create: `packages/sim/pyproject.toml`, `packages/sim/src/sim/**`, `packages/sim/tests/test_import.py`
- Create: `packages/tools/pyproject.toml`, `packages/tools/src/tools/__init__.py`, `packages/tools/tests/test_import.py`

- [ ] **Step 1: Create `packages/sim/pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pact-sim"
version = "0.1.0"
description = "PACT simulation environment (SIL harness, scene generation, digital twin)"
requires-python = ">=3.14"
dependencies = [
    "numpy>=1.24",
    "pact-flight",
]

[tool.hatch.build.targets.wheel]
packages = ["src/sim"]

[tool.uv.sources]
pact-flight = { workspace = true }
```

- [ ] **Step 2: Create the empty `sim` package tree**

Create each as an empty file:

```
packages/sim/src/sim/__init__.py
packages/sim/src/sim/sil/__init__.py
packages/sim/src/sim/scene/__init__.py
packages/sim/src/sim/twin/__init__.py
```

- [ ] **Step 3: Create the `sim` smoke test**

Create `packages/sim/tests/test_import.py`:

```python
"""Smoke test confirming the sim package and its subpackages import."""

import importlib


def test_sim_subpackages_import() -> None:
    """The sim package and its subpackages import without error."""
    for name in ("sim", "sim.sil", "sim.scene", "sim.twin"):
        assert importlib.import_module(name) is not None
```

- [ ] **Step 4: Create `packages/tools/pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pact-tools"
version = "0.1.0"
description = "PACT engineering tools (experiments, training, analysis, replay)"
requires-python = ">=3.14"
dependencies = [
    "pact-flight",
    "pact-sim",
]

[tool.hatch.build.targets.wheel]
packages = ["src/tools"]

[tool.uv.sources]
pact-flight = { workspace = true }
pact-sim = { workspace = true }
```

- [ ] **Step 5: Create the `tools` package and smoke test**

Create empty `packages/tools/src/tools/__init__.py`, then create `packages/tools/tests/test_import.py`:

```python
"""Smoke test confirming the tools package imports."""

import importlib


def test_tools_imports() -> None:
    """The tools package imports without error."""
    assert importlib.import_module("tools") is not None
```

- [ ] **Step 6: Commit**

```bash
git add packages/sim packages/tools
git commit -m "feat(sim,tools): scaffold sim and tools members with smoke tests"
```

---

## Task 4: Add the import-linter layering contracts

**Files:**
- Create: `.importlinter`

- [ ] **Step 1: Create `.importlinter` at the repo root**

```ini
[importlinter]
root_packages =
    flight
    sim
    tools

[importlinter:contract:flight-isolation]
name = Flight must not import sim or tools
type = forbidden
source_modules =
    flight
forbidden_modules =
    sim
    tools

[importlinter:contract:sim-isolation]
name = Sim must not import tools
type = forbidden
source_modules =
    sim
forbidden_modules =
    tools

[importlinter:contract:flight-layers]
name = Flight layered architecture
type = layers
layers =
    flight.core
    flight.payload | flight.thermal | flight.electrical | flight.mechanical | flight.iss_iface | flight.fault
    flight.hal.interfaces
    flight.libs

[importlinter:contract:drivers-from-composition-roots-only]
name = Concrete drivers are reachable only from composition roots
type = forbidden
source_modules =
    flight.payload
    flight.thermal
    flight.electrical
    flight.mechanical
    flight.iss_iface
    flight.fault
    flight.libs
forbidden_modules =
    flight.hal.drivers_real
    flight.hal.drivers_sim
```

Note: the `|` pipes in the `flight-layers` contract declare those six apps as **independent siblings** in one layer -- they may not import one another, and they sit above `hal.interfaces` and `libs` but below `core`. Verification runs in Task 5 (after the packages are installed).

- [ ] **Step 2: Commit**

```bash
git add .importlinter
git commit -m "build: add import-linter dependency-spine contracts"
```

---

## Task 5: Sync the workspace and run all gates green

**Files:** none (verification task)

- [ ] **Step 1: Sync the whole workspace with dev tools**

Run: `uv sync --extra dev`
Expected: resolves and installs the root `pact` package, all three members (`pact-flight`, `pact-sim`, `pact-tools`, installed editable), and the dev tools (pytest, mypy, ruff, import-linter). Exit code 0.

If this fails resolving a heavy **legacy** dependency on Python 3.14 (e.g. `torch`), it is a pre-existing legacy constraint, not introduced here. Contingency: confirm the same `uv sync` succeeds on the developer's machine; if CI is the only failure, align the CI Python in Task 6 to the version with available wheels. Do not remove legacy deps in this phase.

- [ ] **Step 2: Run the test suite (excluding e2e)**

Run: `uv run pytest -m "not e2e" -v`
Expected: PASS, including `test_flight_version_returns_semver`, `test_sim_subpackages_import`, and `test_tools_imports`, alongside the existing `src/pact` tests.

- [ ] **Step 3: Run the type checker on the new packages**

Run: `uv run mypy packages`
Expected: `Success: no issues found`.

- [ ] **Step 4: Run ruff lint and format check**

Run: `uv run ruff check .`
Expected: `All checks passed!`

Run: `uv run ruff format --check packages`
Expected: reports the new files are already formatted (exit code 0). If it reports files would be reformatted, run `uv run ruff format packages`, re-run the check, then `git add packages` and amend nothing -- just include the formatting in the Task's final commit below.

- [ ] **Step 5: Run the import-linter contracts**

Run: `uv run lint-imports`
Expected: `Contracts: 4 kept, 0 broken.` (all contracts pass trivially because the packages are still empty skeletons).

- [ ] **Step 6: Commit the lockfile**

```bash
git add uv.lock
git commit -m "build: sync workspace lockfile"
```

---

## Task 6: Add GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  gates:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.14"
      - name: Sync dependencies
        run: uv sync --extra dev
      - name: Ruff lint
        run: uv run ruff check .
      - name: Ruff format check
        run: uv run ruff format --check packages
      - name: Type check
        run: uv run mypy packages
      - name: Import layering
        run: uv run lint-imports
      - name: Tests
        run: uv run pytest -m "not e2e"
```

- [ ] **Step 2: Re-run the full gate sequence locally to confirm CI will pass**

Run each and confirm the expected output from Task 5:
```bash
uv run ruff check .
uv run ruff format --check packages
uv run mypy packages
uv run lint-imports
uv run pytest -m "not e2e"
```
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions quality gates"
```

---

## Risks & notes

- **Python 3.14 wheels.** Heavy dependencies may lag on py3.14 (the root `pyproject.toml` already documents an albumentations wheel issue). This phase adds only `numpy`, `structlog`, and `import-linter`, all of which have broad wheel support, so Foundation itself is low-risk. The `onnxruntime` recommendation from the spec is a **payload-phase** concern -- verify wheel availability (or choose torch-direct) before that phase, not here.
- **Legacy coexistence.** `src/pact` and the root `pact` package are intentionally left intact. Later phases migrate modules from `src/pact` into `packages/flight/...` and delete `src/pact` only when empty.
- **import-linter config filename.** `.importlinter` is the default file `lint-imports` discovers; no `--config` flag is needed.

---

## Self-review (performed against the spec)

- **Spec coverage (Section 11 tooling + Section 4 layering):** uv workspace (Task 1), lean flight vs heavy tools split via per-member deps (Tasks 2-3), ruff/mypy/pytest (Tasks 1, 5), import-linter contracts for the full dependency spine -- flight isolation, sim isolation, app independence + layering, driver visibility (Task 4), CI (Task 6). The config-default-vs-TOML check (Section 9) is **deferred** to the `libs`/config phase, where the config dataclasses and `config/default.toml` actually exist -- there is nothing to check until then. Noted here so it is not forgotten.
- **Placeholder scan:** no TBD/TODO; every file has complete contents; every command has expected output.
- **Type/name consistency:** import package names `flight`/`sim`/`twin`/etc. are used identically in the package tree, the tests, and the `.importlinter` contracts; `flight_version()` is defined and consumed with the same signature.

---

## Execution notes (applied during the Phase 1 run, 2026-05-30)

The plan executed successfully with these necessary amendments, all confined to the root
`pyproject.toml` and CI config (no `src/pact` changes):

- **Workspace member install:** `uv sync --extra dev` alone did not install the members.
  Added `[tool.uv.sources]` (each member `{ workspace = true }`) and listed `pact-flight` /
  `pact-sim` / `pact-tools` in the `dev` extra so the members install editable.
- **pytest/mypy basename clash:** the two `test_import.py` files collide under pytest's
  default import mode and under mypy. Set `[tool.pytest.ini_options] addopts =
  "--import-mode=importlib"` and `pythonpath = ["."]` (the latter keeps the spawn-based
  legacy test working), and added `namespace_packages = true` + `explicit_package_bases =
  true` to `[tool.mypy]`.
- **uv.lock tracking:** `uv.lock` was gitignored; it is now tracked (removed from
  `.gitignore`) and committed for reproducible CI installs.
- **CI ruff scope:** the `Ruff lint` step is `ruff check packages` (not `.`) because ~286
  pre-existing violations live in legacy `src/`, `tests/`, `scripts/`. Widen to `.` once
  `src/pact` is removed.
- **Deferred to the HAL phase:** add `flight.hal.interfaces` to the
  `drivers-from-composition-roots-only` contract's `source_modules` once the HAL has content.

Verified green locally: pytest (194 passed, plus the 3 new smoke tests), `mypy packages`,
`ruff check packages`, `ruff format --check packages`, `lint-imports` (4/4 kept). Not yet
validated: CI on real Linux py3.14 runners (requires a push).
