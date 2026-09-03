# gse.scenario

**Source:** `packages/gse/src/gse/scenario.py`
**Kind:** module

## Purpose

The scenario module defines the declarative GSE test case model. It loads frozen scenarios
from TOML files for the orchestrator.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SceneSpec` | class | Frame count, seed, thermal and power reading scripts |
| `CommandStep` | class | One telecommand with frame index and sequence |
| `Assertion` | class | Scored or skipped check with id, kind, value, tag |
| `Scenario` | class | Full test case: profile, scene, commands, assertions |
| `load_scenario` | function | Parse a scenario TOML file into a `Scenario` |

## Inputs and outputs

**`load_scenario(path) -> Scenario`**

- Input: filesystem path to scenario TOML.
- Output: frozen `Scenario`.
- Raises `OSError`, `tomllib.TOMLDecodeError`, or `ValidationError` on malformed input.

## Behavior

1. `load_scenario` reads TOML with stdlib `tomllib`.
2. It validates the dict into a frozen `Scenario`. Missing reading arrays default to
   `(20.0,)` thermal and `(10.0,)` power. Missing command and assertion arrays default to
   empty tuples.
3. Assertion tags must be `"frame-portable"` or `"realtime-only"`.
4. It returns the frozen `Scenario` with name, profile, steps, and dt.

## Errors and faults

Raises on missing files, invalid TOML, missing required fields, unknown keys, or unknown
assertion tags. Does not return `Result` (GSE test tooling).

## Messages

None.

## Configuration

Scenario TOML references a profile name (for example `"sil"` or `"sil-link-real"`) applied as
a config override at run time.

## Constraints

- All dataclasses are frozen and hashable.
- `SimScalarSensor` holds the last reading once a script exhausts.
- A hot thermal reading publishes `thermal_sample` telemetry and does not emit
  `THERMAL_OVER_LIMIT`.

## Related documents

- [`gse`](gse.md)
- [`gse.orchestrator`](orchestrator.md)
- [`gse.harness`](harness.md)
