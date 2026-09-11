"""Scenario model + loader for the GSE deterministic harness.

A Scenario is a fully declarative test case: which profile to wire, what scene to render,
which commands to inject at which frame, and which assertions to score. Assertions carry a
tag: "frame-portable" assertions hold under the deterministic in-process backend (mode,
ack status, gimbal motion, counts) and are scored; "realtime-only" assertions (e.g. wall-clock
ack latency) are DEFINED here but recorded skipped-with-reason under the in-process backend.
Scenarios are loaded from TOML via tomllib (stdlib) and validated into frozen dataclasses.

Contains:
  - SceneSpec: which scene to render (num_frames, seed).
  - CommandStep: one command to inject at a given frame index.
  - Assertion: one scored/skipped check (id, kind, value, frame-portable|realtime-only tag).
  - Scenario: the whole declarative case.
  - load_scenario: parse a scenario TOML file into a Scenario.

Satisfies: REQ-VAL-GSE-001.
"""

from __future__ import annotations

# stdlib
import tomllib
from typing import Literal

# third-party
from pydantic import ConfigDict, Field, TypeAdapter
from pydantic.dataclasses import dataclass

ParamValue = str | int | float | bool
AssertionTag = Literal["frame-portable", "realtime-only"]

_SCHEMA = ConfigDict(extra="forbid")


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class SceneSpec:
    """Which deterministic scene the harness renders for a scenario.

    Fields:
        num_frames: Number of mosaic frames to render (one per SIL step).
        seed: Deterministic render seed.
        thermal_readings: Per-step thermal-sensor readings (deg C) the SimScalarSensor serves;
            a SimScalarSensor holds its last value once exhausted, so a singleton drives a
            constant temperature for the whole run. Defaults to (20.0,) (nominal). A hot
            reading publishes thermal_sample telemetry and does not emit THERMAL_OVER_LIMIT.
        power_readings: Per-step power-sensor readings (W), same hold-last semantics. Defaults
            to (10.0,) (nominal).
    """

    num_frames: int
    seed: int
    thermal_readings: tuple[float, ...] = (20.0,)
    power_readings: tuple[float, ...] = (10.0,)


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class CommandStep:
    """One telecommand to inject at a given frame index during a scenario run.

    Fields:
        at_frame: 1-based step index at which the command is injected.
        command_id: The command opcode string (e.g. "SET_THERMAL_LIMIT", "PING").
        params: The command parameter dict.
        source: The command origin identifier (must be on the flight allow-list to accept).
        seq: The per-source monotonic sequence number.
    """

    at_frame: int
    command_id: str
    source: str
    seq: int
    params: dict[str, ParamValue] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class Assertion:
    """One scenario assertion, scored or skipped depending on its tag.

    Fields:
        id: Stable identifier for the assertion (cited as evidence in the VCRM).
        kind: The assertion kind ("mode_is", "command_acked", "gimbal_moved",
            "min_inference_count", "min_downlink_count", "ack_within_seconds").
        value: The expected value (kind-dependent: a mode/status string, a bool, an int,
            or a float seconds budget).
        tag: "frame-portable" (scored under the in-process backend) or "realtime-only"
            (recorded skipped-with-reason under the in-process backend).
    """

    id: str
    kind: str
    value: ParamValue
    tag: AssertionTag


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class Scenario:
    """A fully declarative GSE test case: profile + scene + commands + assertions.

    Fields:
        name: Human-readable scenario name (also the evidence id stem in the VCRM).
        profile: Profile name applied as a load_config override (e.g. "sil", "sil-link-real").
        scene: The SceneSpec to render.
        commands: The telecommands to inject, in declaration order.
        assertions: The assertions to score/skip, in declaration order.
        steps: Number of deterministic steps to run.
        dt: Seconds to advance per step.
    """

    name: str
    profile: str
    scene: SceneSpec
    steps: int
    dt: float
    commands: tuple[CommandStep, ...] = ()
    assertions: tuple[Assertion, ...] = ()


_SCENARIO_ADAPTER = TypeAdapter(Scenario)


def load_scenario(path: str) -> Scenario:
    """Parse a scenario TOML file into a typed, frozen Scenario.

    Args:
        path: Filesystem path to the scenario TOML file.

    Returns:
        The parsed Scenario.

    Raises:
        OSError: if the file cannot be read.
        tomllib.TOMLDecodeError: if the file is not valid TOML.
        ValidationError: if a required field is missing, a tag is unknown, or a
            value fails the schema.

    Notes:
        GSE test tooling, so this raises on malformed input rather than returning a Result.
        commands/assertions are normalized to tuples so the returned Scenario is fully frozen.
        Each assertion's tag is taken verbatim from the TOML ("frame-portable"
        or "realtime-only") and is the only signal the orchestrator uses to score-vs-skip it.
    """
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    return _SCENARIO_ADAPTER.validate_python(data)
