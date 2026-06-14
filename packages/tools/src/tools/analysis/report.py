"""Static report bundle emitter: per-run data + figures + summary (md/html) + manifest.

Writes one self-contained bundle per captured scenario run -- tidy datasets (the long frame and
every per-group wide frame, each as Parquet + CSV, plus the summary-stats table), a figures/ tree
(one subdirectory per group), a human-readable summary (Markdown + standalone HTML), and a
machine-readable manifest.json -- and a suite-level index over many runs. No wall-clock timestamps
are written, so a bundle is byte-reproducible from the deterministic capture.

Contains:
  - RunReport / SuiteReport: the paths + manifest produced for one run / a suite.
  - write_run_report: emit one run's full bundle.
  - write_suite_report: run-report every scenario and emit a suite index.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

# stdlib
import html
import json
from dataclasses import dataclass
from pathlib import Path

# third-party
import pandas as pd

# internal
from tools.analysis.datapoints import REGISTRY, accumulable_names
from tools.analysis.plots import PLOT_GROUPS, build_group_figures
from tools.analysis.plots.common import save_figures
from tools.analysis.recorder import CaptureResult
from tools.analysis.runner import ScenarioRun
from tools.analysis.stats import summarize

SCHEMA_VERSION = 1
_STATS_VIEW = ("signal", "kind", "unit", "last", "min", "max", "mean", "n_transitions", "mode")


@dataclass(frozen=True, slots=True)
class RunReport:
    """The artifacts written for one scenario run."""

    name: str
    out_dir: Path
    n_figures: int
    manifest: dict[str, object]


@dataclass(frozen=True, slots=True)
class SuiteReport:
    """The artifacts written for a whole suite (index + per-run reports)."""

    suite: str
    out_dir: Path
    runs: tuple[RunReport, ...]


def _write_frame(frame: pd.DataFrame, stem: Path) -> None:
    """Write a frame as both Parquet and CSV at ``<stem>.parquet`` / ``<stem>.csv``."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(f"{stem}.parquet", engine="pyarrow", index=False)
    frame.to_csv(f"{stem}.csv", index=False)


def _fmt(value: object) -> str:
    """Format a stats cell for a text table (compact floats, blanks for NaN/empty)."""
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.4g}"
    return str(value)


def _outcomes(capture: CaptureResult) -> dict[str, object]:
    """Derive a few headline facts from a capture for quick scanning / the manifest."""

    def last(group: str, column: str) -> object:
        frame = capture.wide.get(group)
        if frame is None or column not in frame.columns:
            return None
        return _native(frame[column].iloc[-1])

    def ever_positive(group: str, column: str) -> bool:
        frame = capture.wide.get(group)
        if frame is None or column not in frame.columns:
            return False
        return bool((pd.to_numeric(frame[column], errors="coerce") > 0).any())

    return {
        "safe_latched_end": bool(last("system", "system.safe_latched") or 0),
        "safe_ever": ever_positive("system", "system.safe_latched"),
        "final_gimbal_state": last("payload", "payload.gimbal_state"),
        "final_system_mode": last("system", "system.mode"),
        "stow_engaged_ever": ever_positive("payload", "payload.stow_switch"),
        "total_faults": _native(pd.to_numeric(capture.wide["system"]["system.total_faults"]).sum()),
        "final_model_deploy_state": last("model_deploy", "model_deploy.state"),
        "storage_entries_evicted": last("storage", "storage.dropped_count"),
        "downlink_pending_peak": _native(
            pd.to_numeric(capture.wide["downlink"]["downlink.pending_items"]).max()
        ),
        "final_launch_lock_state": last("mechanical", "mechanical.launch_lock_state"),
    }


def _native(value: object) -> object:
    """Coerce a numpy/pandas scalar to a JSON-native Python scalar."""
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    return item() if callable(item) else str(value)


def write_run_report(run: ScenarioRun, out_dir: Path) -> RunReport:
    """Emit the full static bundle for one captured scenario run.

    Args:
        run: the captured ScenarioRun.
        out_dir: the bundle root (created if needed); receives data/, figures/, summary.md,
            summary.html, and manifest.json.

    Returns:
        A RunReport with the bundle root, figure count, and the manifest dict.
    """
    capture = run.capture
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = summarize(capture)

    data_dir = out_dir / "data"
    _write_frame(capture.long, data_dir / "long")
    _write_frame(stats, data_dir / "stats")
    for group, frame in capture.wide.items():
        _write_frame(frame.reset_index(), data_dir / "wide" / group)

    figures_dir = out_dir / "figures"
    group_figures: list[tuple[str, list[tuple[str, str, str]]]] = []
    total_figures = 0
    for entry in PLOT_GROUPS:
        figures = build_group_figures(entry.group, capture.wide[entry.group])
        save_figures(figures, figures_dir / entry.group)
        rels = [
            (figure.name, figure.title, f"figures/{entry.group}/{figure.name}.png")
            for figure in figures
        ]
        group_figures.append((entry.group, rels))
        total_figures += len(rels)

    manifest = _build_manifest(run, capture, group_figures, total_figures)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(
        _render_markdown(run, stats, group_figures, manifest), encoding="utf-8"
    )
    (out_dir / "summary.html").write_text(
        _render_html(run, stats, group_figures, manifest), encoding="utf-8"
    )
    return RunReport(run.spec.name, out_dir, total_figures, manifest)


def _build_manifest(
    run: ScenarioRun,
    capture: CaptureResult,
    group_figures: list[tuple[str, list[tuple[str, str, str]]]],
    total_figures: int,
) -> dict[str, object]:
    """Assemble the machine-readable manifest for one run (no wall-clock fields; deterministic)."""
    spec = run.spec
    groups: dict[str, object] = {}
    for group, rels in group_figures:
        groups[group] = {
            "n_signals": sum(1 for signal in REGISTRY if signal.group == group),
            "n_figures": len(rels),
            "figures": [name for name, _title, _rel in rels],
        }
    return {
        "tool": "tools.analysis",
        "schema_version": SCHEMA_VERSION,
        "deterministic": True,
        "scenario": {
            "name": spec.name,
            "title": spec.title,
            "description": spec.description,
            "category": spec.category,
            "steps": spec.steps,
            "dt": spec.dt,
            "num_frames": spec.frame_count(),
            "seed": spec.seed,
        },
        "capture": {
            "n_steps": capture.n_steps,
            "n_signals": capture.n_signals,
            "n_columns": capture.n_columns,
            "groups": list(capture.wide.keys()),
        },
        "datapoints": {
            "registry_signals": len(REGISTRY),
            "cumulative_columns": len(accumulable_names()),
            "total_columns": capture.n_columns,
        },
        "groups": groups,
        "total_figures": total_figures,
        "data_files": [
            "data/long.parquet",
            "data/long.csv",
            "data/stats.parquet",
            "data/stats.csv",
            *[f"data/wide/{group}.parquet" for group in capture.wide],
            *[f"data/wide/{group}.csv" for group in capture.wide],
        ],
        "outcomes": _outcomes(capture),
    }


def _stats_for_group(stats: pd.DataFrame, group: str) -> pd.DataFrame:
    """Return the compact stats view for one group (the columns in _STATS_VIEW)."""
    subset = stats[stats["group"] == group]
    return subset[list(_STATS_VIEW)]


def _md_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored Markdown table."""
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(_fmt(value) for value in record) + " |"
        for record in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _render_markdown(
    run: ScenarioRun,
    stats: pd.DataFrame,
    group_figures: list[tuple[str, list[tuple[str, str, str]]]],
    manifest: dict[str, object],
) -> str:
    """Render the human-readable Markdown summary for one run."""
    spec = run.spec
    datapoints = manifest["datapoints"]
    assert isinstance(datapoints, dict)
    lines = [
        f"# SIL Analysis: {spec.title}",
        "",
        spec.description,
        "",
        "## Run parameters",
        "",
        f"- **scenario**: `{spec.name}` ({spec.category})",
        f"- **steps**: {spec.steps} at dt = {spec.dt}s; **frames**: {spec.frame_count()}",
        f"- **datapoints**: {datapoints['registry_signals']} registry signals "
        f"+ {datapoints['cumulative_columns']} cumulative = "
        f"{datapoints['total_columns']} columns x {run.capture.n_steps} steps",
        "",
        "## Outcomes",
        "",
    ]
    outcomes = manifest["outcomes"]
    assert isinstance(outcomes, dict)
    for key, value in outcomes.items():
        lines.append(f"- **{key}**: {_fmt(value)}")
    lines.append("")
    lines.append("## Per-group time series + summary statistics")
    lines.append("")
    for group, rels in group_figures:
        lines.append(f"### {group}")
        lines.append("")
        for _name, title, rel in rels:
            lines.append(f"![{title}]({rel})")
            lines.append("")
        lines.append(_md_table(_stats_for_group(stats, group)))
        lines.append("")
    return "\n".join(lines)


def _render_html(
    run: ScenarioRun,
    stats: pd.DataFrame,
    group_figures: list[tuple[str, list[tuple[str, str, str]]]],
    manifest: dict[str, object],
) -> str:
    """Render a standalone HTML summary (embedded relative figure links + stats tables)."""
    spec = run.spec
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>SIL Analysis: {html.escape(spec.title)}</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:1100px}"
        "img{max-width:100%;border:1px solid #ddd;margin:.4rem 0}"
        "table{border-collapse:collapse;font-size:.85rem;margin:.6rem 0}"
        "td,th{border:1px solid #ccc;padding:2px 6px;text-align:right}"
        "th{background:#f3f3f3}h2{border-top:2px solid #eee;padding-top:.6rem}</style>",
        "</head><body>",
        f"<h1>SIL Analysis: {html.escape(spec.title)}</h1>",
        f"<p>{html.escape(spec.description)}</p>",
        f"<p><b>scenario</b> <code>{html.escape(spec.name)}</code> ({html.escape(spec.category)}); "
        f"<b>steps</b> {spec.steps} at dt={spec.dt}s; "
        f"<b>columns</b> {run.capture.n_columns} x {run.capture.n_steps} steps</p>",
        "<h2>Outcomes</h2><ul>",
    ]
    outcomes = manifest["outcomes"]
    assert isinstance(outcomes, dict)
    for key, value in outcomes.items():
        parts.append(f"<li><b>{html.escape(key)}</b>: {html.escape(_fmt(value))}</li>")
    parts.append("</ul>")
    for group, rels in group_figures:
        parts.append(f"<h2>{html.escape(group)}</h2>")
        for _name, title, rel in rels:
            parts.append(
                f'<figure><img src="{rel}" alt="{html.escape(title)}">'
                f"<figcaption>{html.escape(title)}</figcaption></figure>"
            )
        parts.append(_stats_for_group(stats, group).to_html(index=False, na_rep=""))
    parts.append("</body></html>")
    return "\n".join(parts)


def write_suite_report(runs: list[ScenarioRun], suite: str, out_dir: Path) -> SuiteReport:
    """Emit a per-run bundle for every run plus a suite-level index (md/html/manifest).

    Args:
        runs: the captured scenario runs.
        suite: the suite name (recorded in the index manifest).
        out_dir: the suite bundle root; each run lands in ``out_dir/<scenario>``.

    Returns:
        A SuiteReport with the per-run RunReports.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = tuple(write_run_report(run, out_dir / run.spec.name) for run in runs)
    index_manifest = {
        "tool": "tools.analysis",
        "schema_version": SCHEMA_VERSION,
        "suite": suite,
        "n_runs": len(reports),
        "runs": [
            {
                "name": report.name,
                "title": runs[position].spec.title,
                "category": runs[position].spec.category,
                "n_figures": report.n_figures,
                "outcomes": report.manifest["outcomes"],
            }
            for position, report in enumerate(reports)
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(index_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_suite_index(runs, reports, suite, out_dir)
    return SuiteReport(suite, out_dir, reports)


def _write_suite_index(
    runs: list[ScenarioRun], reports: tuple[RunReport, ...], suite: str, out_dir: Path
) -> None:
    """Write the suite index.md + index.html linking to each run's summary."""
    md = [
        f"# SIL Analysis Suite: `{suite}`",
        "",
        f"{len(reports)} scenario runs.",
        "",
        "| scenario | category | figures | SAFE ever | final state |",
        "| --- | --- | --- | --- | --- |",
    ]
    rows_html = []
    for position, report in enumerate(reports):
        spec = runs[position].spec
        outcomes = report.manifest["outcomes"]
        assert isinstance(outcomes, dict)
        safe = _fmt(outcomes.get("safe_ever"))
        final = _fmt(outcomes.get("final_gimbal_state"))
        md.append(
            f"| [{spec.name}]({spec.name}/summary.md) | {spec.category} | "
            f"{report.n_figures} | {safe} | {final} |"
        )
        rows_html.append(
            f'<tr><td><a href="{spec.name}/summary.html">{html.escape(spec.name)}</a></td>'
            f"<td>{html.escape(spec.category)}</td><td>{report.n_figures}</td>"
            f"<td>{html.escape(safe)}</td><td>{html.escape(final)}</td></tr>"
        )
    (out_dir / "index.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    html_doc = [
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">',
        f"<title>SIL Analysis Suite: {html.escape(suite)}</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:2rem}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}</style>",
        "</head><body>",
        f"<h1>SIL Analysis Suite: {html.escape(suite)}</h1>",
        f"<p>{len(reports)} scenario runs.</p>",
        "<table><tr><th>scenario</th><th>category</th><th>figures</th>"
        "<th>SAFE ever</th><th>final state</th></tr>",
        *rows_html,
        "</table></body></html>",
    ]
    (out_dir / "index.html").write_text("\n".join(html_doc), encoding="utf-8")
