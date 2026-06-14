"""PACT SIL telemetry capture, analysis, and static-report tooling.

A turnkey, read-only observability tool: it drives the deterministic SIL (the real flight apps
over sim drivers), passively captures per-step datapoints across all ten flight apps plus the
message bus, and emits a static report bundle per run (matplotlib figures, a Markdown/HTML
summary, and tidy Parquet/CSV datasets). It touches the flight software only through read-only
accessors (the bus queue_depth() hook and read-only app-state introspection); it never changes
flight behavior, never publishes telemetry, and never adds a control-flow path.

Run: ``python -m tools.analysis run <suite|scenario> --out artifacts/analysis/<run>``.

Contains (subpackage modules):
  - datapoints: the typed per-step signal registry + SampleContext/DeviceSample.
  - recorder: the passive capture loop (owns step_once; builds tidy long + per-app wide frames).
  - runner: ScenarioSpec + the deterministic scenario builders.
  - characterize: run a named suite of scenarios.
  - stats: per-signal summary statistics.
  - plots: per-app + bus matplotlib figure builders.
  - report: emit the per-run bundle (data, figures, summary.md/.html, manifest.json).
  - cli: the ``run`` command-line entry point.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations
