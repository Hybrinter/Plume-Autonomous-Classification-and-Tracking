# gse.orchestrator

**Source:** `packages/gse/src/gse/orchestrator.py`
**Kind:** module

## Purpose

The orchestrator runs a scenario through a harness backend and scores its assertions. It
emits JSON V&V evidence records for validation runs.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `AssertionResult` | class | Per-assertion pass, fail, or skip outcome |
| `ScenarioReport` | class | Rolled-up pass, fail, and skip counts |
| `run_scenario` | function | Drive scenario end-to-end and score assertions |
| `scenario_report_to_json` | function | Serialize a report to JSON |
| `write_evidence_record` | function | Write `<scenario>.vv.json` to a directory |

## Inputs and outputs

**`run_scenario(scenario, profile_path, backend=None) -> ScenarioReport`**

- Inputs: frozen `Scenario`, profile TOML path, optional `HarnessBackend` (defaults to fresh
  `InProcessBackend`).
- Output: `ScenarioReport` with ordered `AssertionResult` entries.

**`scenario_report_to_json(report) -> str`**

- Output: key-sorted indented JSON string.

**`write_evidence_record(report, directory) -> str`**

- Output: path of the written evidence file.

## Behavior

1. `run_scenario` calls `backend.build(scenario, profile_path)`.
2. For each frame 1 through `scenario.steps`, it injects commands whose `at_frame` matches,
   advances `now` by `scenario.dt`, and calls `backend.step`.
3. It calls `backend.collect()` and always calls `backend.shutdown()` in a `finally` block.
4. Frame-portable assertions score against the capture. Realtime-only assertions record
   status `"skip"` with a fixed reason.
5. `_score_frame_portable` evaluates kinds: `mode_is`, `command_acked`, `gimbal_moved`,
   `min_inference_count`, `min_downlink_count`.
6. Evidence helpers write deterministic JSON without wall-clock timestamps.

## Errors and faults

Scoring errors still trigger backend shutdown via `finally`. File I/O raises normally (GSE
test tooling, not flight library code).

## Messages

None directly. Scoring reads drained message summaries inside `TelemetryCapture`.

## Configuration

Profile path selects environment via `load_profile_config` inside the backend.

## Constraints

- Realtime-only assertions (for example `ack_within_seconds`) are skipped under the
  in-process backend.
- `mode_is` treats NOMINAL as "no SAFE published" and SAFE as "at least one SAFE seen".
- Sim-link command timing does not affect scoring when commands pre-ingest on step 1.

## Related documents

- [`gse`](gse.md)
- [`gse.harness`](harness.md)
- [`gse.scenario`](scenario.md)
