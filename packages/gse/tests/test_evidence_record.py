"""JSON V&V evidence-record emission for scored scenarios (ADR-0010 Decision 3)."""

from __future__ import annotations

import json
from pathlib import Path

from gse.orchestrator import (
    AssertionResult,
    ScenarioReport,
    scenario_report_to_json,
    write_evidence_record,
)


def _report() -> ScenarioReport:
    """A small two-assertion report (one pass, one skip) for serialization tests."""
    return ScenarioReport(
        scenario="demo",
        passed=1,
        failed=0,
        skipped=1,
        results=(
            AssertionResult(id="a", tag="frame-portable", status="pass", detail="ok"),
            AssertionResult(id="b", tag="realtime-only", status="skip", detail="skipped"),
        ),
    )


def test_scenario_report_to_json_captures_counts_and_results() -> None:
    """The evidence record is valid JSON capturing the counts and every assertion result."""
    record = json.loads(scenario_report_to_json(_report()))
    assert record["scenario"] == "demo"
    assert record["passed"] == 1
    assert record["failed"] == 0
    assert record["skipped"] == 1
    assert {r["id"] for r in record["results"]} == {"a", "b"}
    assert {r["status"] for r in record["results"]} == {"pass", "skip"}


def test_write_evidence_record_writes_reloadable_file(tmp_path: Path) -> None:
    """write_evidence_record emits <scenario>.vv.json that reloads to the same record."""
    path = write_evidence_record(_report(), str(tmp_path))
    assert Path(path).name == "demo.vv.json"
    reloaded = json.loads(Path(path).read_text(encoding="utf-8"))
    assert reloaded["scenario"] == "demo"
    assert len(reloaded["results"]) == 2
