# tools.analysis.characterize

**Source:** `packages/tools/src/tools/analysis/characterize.py`
**Kind:** module

## Purpose

Characterize resolves a suite or single scenario name to `ScenarioSpec` lists and captures
every run. It is the orchestration layer above the runner.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SUITES` | constant | Named groupings of built-in scenario names |
| `suite_names` | function | Return every selectable suite name |
| `repo_root` | function | Locate repository root via `scenarios/` + `pyproject.toml` |
| `scenario_file_paths` | function | Sorted `scenarios/*.toml` paths |
| `suite_specs` | function | Resolve a name to an ordered spec list |
| `run_suite` | function | Capture every spec in a resolved suite |

## Inputs and outputs

**`suite_specs(name) -> list[ScenarioSpec]`**

- Accepts `"full"`, `"builtin"`, `"files"`, a key in `SUITES`, or a built-in scenario name.
- Raises `KeyError` for unknown names.

**`run_suite(name) -> list[ScenarioRun]`**

- Output: one `ScenarioRun` per captured scenario.

## Behavior

1. `"full"` runs every built-in scenario plus every repo scenario TOML file.
2. `"builtin"` runs built-in scenarios only.
3. `"files"` adapts each `scenarios/*.toml` via `load_scenario_spec`.
4. Named suites (`smoke`, `faults`, `commands`, `resources`) run fixed name lists.
5. A single built-in scenario name resolves to a one-element spec list.
6. `run_suite` calls `run_scenario` for each spec.

## Errors and faults

`repo_root` raises `RuntimeError` when no ancestor has both `scenarios/` and `pyproject.toml`.

## Messages

None.

## Configuration

None directly. Specs may carry custom `PactConfig` overrides.

## Constraints

- Meta-suite names are fixed: `full`, `builtin`, `files`.
- File scenarios ignore GSE assertions. Capture is passive.

## Related documents

- [`tools.analysis`](analysis.md)
- [`tools.analysis.runner`](runner.md)
- [`tools.analysis.cli`](cli.md)
