"""GSE scenario orchestrator: run a scenario through a backend and score its assertions.

run_scenario builds the scenario on a HarnessBackend (default InProcessBackend), steps it
through its frame timeline, injects each command at its at_frame, collects a
TelemetryCapture, and scores every assertion. Frame-portable assertions are evaluated
against the deterministic capture; realtime-only assertions are recorded status="skip"
with a fixed reason (they are not meaningful under a ManualClock-driven in-process
backend -- they are reserved for the PIL/HIL socket backends).

Contains:
  - AssertionResult: per-assertion pass/fail/skip outcome with a detail string.
  - ScenarioReport: rolled-up counts + the ordered per-assertion results.
  - run_scenario: drive a scenario end-to-end and score it into a ScenarioReport.

Satisfies: REQ-COMM-HIGH-001, REQ-COMM-HIGH-003, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass
from typing import Literal

# internal
from flight.libs.types import AckStatus, SystemMode

from gse.harness import HarnessBackend, InProcessBackend, TelemetryCapture
from gse.scenario import Assertion, Scenario

_SKIP_REASON = "realtime-only: not evaluated under deterministic in-process backend"
_FRAME_PORTABLE_KINDS = frozenset(
    {"mode_is", "command_acked", "gimbal_moved", "min_inference_count", "min_downlink_count"}
)


@dataclass(frozen=True, slots=True)
class AssertionResult:
    """Outcome of scoring one scenario assertion.

    Fields:
        id: The assertion's stable identifier (echoed from the scenario).
        tag: "frame-portable" or "realtime-only" (echoed from the scenario).
        status: "pass", "fail", or "skip".
        detail: Human-readable explanation (expected vs. observed, or the skip reason).
    """

    id: str
    tag: str
    status: Literal["pass", "fail", "skip"]
    detail: str


@dataclass(frozen=True, slots=True)
class ScenarioReport:
    """Rolled-up scoring report for one scenario run.

    Fields:
        scenario: The scenario name.
        passed: Count of frame-portable assertions that passed.
        failed: Count of frame-portable assertions that failed.
        skipped: Count of realtime-only assertions recorded as skipped.
        results: The ordered per-assertion results (one per scenario assertion).
    """

    scenario: str
    passed: int
    failed: int
    skipped: int
    results: tuple[AssertionResult, ...]


def _score_frame_portable(assertion: Assertion, capture: TelemetryCapture) -> AssertionResult:
    """Score one frame-portable assertion against a deterministic TelemetryCapture.

    Args:
        assertion: A frame-portable assertion (kind in _FRAME_PORTABLE_KINDS).
        capture: The collected telemetry to evaluate against.

    Returns:
        AssertionResult: pass/fail with an expected-vs-observed detail string.

    Notes:
        - mode_is: the scenario's terminal mode expectation. NOMINAL is satisfied iff NO
          SAFE was ever published; SAFE is satisfied iff at least one SAFE was published.
        - command_acked: an ACCEPTED/REJECTED ack of that status must appear in the run.
        - gimbal_moved: matches capture.gimbal_moved against the expected bool.
        - min_inference_count / min_downlink_count: observed >= the integer floor.
    """
    kind = assertion.kind
    if kind == "gimbal_moved":
        expected = bool(assertion.value)
        ok = capture.gimbal_moved == expected
        detail = f"expected gimbal_moved={expected}, got {capture.gimbal_moved}"
        return _result(assertion, ok, detail)
    if kind == "min_inference_count":
        floor = int(assertion.value)
        ok = capture.inference_count >= floor
        detail = f"inference_count {capture.inference_count} >= {floor}"
        return _result(assertion, ok, detail)
    if kind == "min_downlink_count":
        floor = int(assertion.value)
        observed = len(capture.downlink_packets)
        ok = observed >= floor
        detail = f"downlink_count {observed} >= {floor}"
        return _result(assertion, ok, detail)
    if kind == "command_acked":
        expected_status = AckStatus[str(assertion.value)]
        ok = expected_status in capture.acks
        detail = f"expected ack {expected_status.value} in {list(capture.acks)}"
        return _result(assertion, ok, detail)
    if kind == "mode_is":
        expected_mode = SystemMode[str(assertion.value)]
        saw_safe = SystemMode.SAFE in capture.mode_changes
        ok = saw_safe if expected_mode is SystemMode.SAFE else not saw_safe
        detail = f"expected mode {expected_mode.value}, safe_seen={saw_safe}"
        return _result(assertion, ok, detail)
    return _result(assertion, False, f"unknown frame-portable kind {kind!r}")


def _result(assertion: Assertion, ok: bool, detail: str) -> AssertionResult:
    """Build a pass/fail AssertionResult echoing the assertion id + tag."""
    return AssertionResult(
        id=assertion.id,
        tag=assertion.tag,
        status="pass" if ok else "fail",
        detail=detail,
    )


def run_scenario(
    scenario: Scenario,
    profile_path: str,
    backend: HarnessBackend | None = None,
) -> ScenarioReport:
    """Run a scenario through a backend and score every assertion into a ScenarioReport.

    Args:
        scenario: The scenario (scene spec, command timeline, assertions, steps, dt).
        profile_path: Path to the profile TOML override (selects the per-axis environment).
        backend: The HarnessBackend to run on; defaults to a fresh InProcessBackend.

    Returns:
        ScenarioReport: pass/fail/skip counts and the ordered per-assertion results.

    Notes:
        Steps the backend over scenario.steps cycles at scenario.dt seconds, injecting each
        command at its at_frame (1-based: injected just before the step that processes that
        frame). On a sim link inject_command is a no-op (commands are pre-baked and all
        ingest on step 1), so at_frame timing only takes effect on a real link. Frame-portable
        assertions are scored against the collected capture; realtime-only assertions are
        recorded status="skip" with a fixed reason. The backend is always shut down, even on a
        scoring error.
    """
    runner = backend if backend is not None else InProcessBackend()
    runner.build(scenario, profile_path)
    try:
        now = 0.0
        for frame in range(1, scenario.steps + 1):
            for step in scenario.commands:
                if step.at_frame == frame:
                    runner.inject_command(step)
            now += scenario.dt
            runner.step(now)
        capture = runner.collect()
    finally:
        runner.shutdown()

    results: list[AssertionResult] = []
    for assertion in scenario.assertions:
        if assertion.tag == "realtime-only" or assertion.kind not in _FRAME_PORTABLE_KINDS:
            results.append(
                AssertionResult(
                    id=assertion.id, tag=assertion.tag, status="skip", detail=_SKIP_REASON
                )
            )
            continue
        results.append(_score_frame_portable(assertion, capture))

    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    skipped = sum(1 for r in results if r.status == "skip")
    return ScenarioReport(
        scenario=scenario.name,
        passed=passed,
        failed=failed,
        skipped=skipped,
        results=tuple(results),
    )
