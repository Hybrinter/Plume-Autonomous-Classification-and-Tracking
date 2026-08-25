# tools.analysis

**Source:** `packages/tools/src/tools/analysis/`
**Kind:** package

## Purpose

The analysis package captures the deterministic SIL passively and emits static report bundles.
Each run produces datasets, matplotlib figures, summary pages, and a manifest.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`__main__`](analysis/__main__.md) | module | `python -m tools.analysis` entry dispatch |
| [`cli`](analysis/cli.md) | module | `run` and `list` commands |
| [`characterize`](analysis/characterize.md) | module | Named scenario suites |
| [`runner`](analysis/runner.md) | module | Built-in scenario specs and SIL wiring |
| [`recorder`](analysis/recorder.md) | module | Passive per-step capture loop |
| [`datapoints`](analysis/datapoints.md) | module | Typed per-step signal registry |
| [`stats`](analysis/stats.md) | module | Per-signal summary statistics |
| [`report`](analysis/report.md) | module | Bundle writer (data, figures, summaries) |
| [`plots`](analysis/plots.md) | package | Per-group matplotlib figure builders |

## Package interface

`tools.analysis.__init__` carries module docstring only. The CLI entry is
`python -m tools.analysis`.

## Interactions

Analysis imports `sim.scene` and `sim.sil`. It drives `build_sil_system`, owns the
`step_once` loop in `recorder`, and subscribes to nineteen bus message types.

It reads app state and sim driver fields read-only. It never publishes telemetry or changes
control flow.

## Constraints

- Capture is fully passive and deterministic. Re-running reproduces identical bundles.
- Extractor failures record NaN (numeric) or "" (categorical) in the output frames.
- No wall-clock timestamps appear in manifests or evidence files.

## Related documents

- [`tools`](tools.md)
- [`tools.analysis.cli`](analysis/cli.md)
- [`tools.analysis.recorder`](analysis/recorder.md)
- [`sim.sil`](sim/sil.md)
