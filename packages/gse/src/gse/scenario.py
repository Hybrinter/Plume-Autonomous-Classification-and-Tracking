"""Scenario model + loader for the GSE deterministic harness.

A Scenario is a fully declarative test case: which profile to wire, what scene to render,
which commands to inject at which frame, and which assertions to score. Assertions carry a
tag: "frame-portable" assertions hold under the deterministic in-process backend (mode,
ack status, gimbal motion, counts) and are scored; "realtime-only" assertions (e.g. wall-clock
ack latency) are DEFINED here but recorded skipped-with-reason under the in-process backend.
Scenarios are loaded from TOML via tomllib (stdlib). All dataclasses are frozen.

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
from dataclasses import dataclass
from typing import Literal

ParamValue = str | int | float | bool
AssertionTag = Literal["frame-portable", "realtime-only"]


@dataclass(frozen=True, slots=True)
class SceneSpec:
    """Which deterministic scene the harness renders for a scenario.

    Fields:
        num_frames: Number of mosaic frames to render (one per SIL step).
        seed: Deterministic render seed.
        thermal_readings: Per-step thermal-sensor readings (deg C) the SimScalarSensor serves;
            a SimScalarSensor holds its last value once exhausted, so a singleton drives a
            constant temperature for the whole run. Defaults to (20.0,) (nominal). A reading
            above config.fault.thermal_limit_c drives THERMAL_OVER_LIMIT -> SAFE.
        power_readings: Per-step power-sensor readings (W), same hold-last semantics. Defaults
            to (10.0,) (nominal).
    """

    num_frames: int
    seed: int
    thermal_readings: tuple[float, ...] = (20.0,)
    power_readings: tuple[float, ...] = (10.0,)


@dataclass(frozen=True, slots=True)
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
    params: dict[str, ParamValue]
    source: str
    seq: int


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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
    commands: tuple[CommandStep, ...]
    assertions: tuple[Assertion, ...]
    steps: int
    dt: float


def load_scenario(path: str) -> Scenario:
    """Parse a scenario TOML file into a typed, frozen Scenario.

    Args:
        path: Filesystem path to the scenario TOML file.

    Returns:
        The parsed Scenario.

    Raises:
        OSError: if the file cannot be read.
        tomllib.TOMLDecodeError: if the file is not valid TOML.
        KeyError: if a required scenario/scene/command/assertion field is missing.

    Notes:
        GSE test tooling, so this raises on malformed input rather than returning a Result.
        commands/assertions are normalized to tuples so the returned Scenario is fully frozen
        and hashable. Each assertion's tag is taken verbatim from the TOML ("frame-portable"
        or "realtime-only") and is the only signal the orchestrator uses to score-vs-skip it.
    """
    with open(path, "rb") as handle:
        data = tomllib.load(handle)

    scene_raw = data["scene"]
    scene = SceneSpec(
        num_frames=int(scene_raw["num_frames"]),
        seed=int(scene_raw["seed"]),
        thermal_readings=_readings(scene_raw.get("thermal_readings"), (20.0,)),
        power_readings=_readings(scene_raw.get("power_readings"), (10.0,)),
    )

    commands = tuple(
        CommandStep(
            at_frame=int(cmd["at_frame"]),
            command_id=str(cmd["command_id"]),
            params=dict(cmd.get("params", {})),
            source=str(cmd["source"]),
            seq=int(cmd["seq"]),
        )
        for cmd in data.get("commands", [])
    )

    assertions = tuple(
        Assertion(
            id=str(item["id"]),
            kind=str(item["kind"]),
            value=item["value"],
            tag=_parse_tag(item["tag"]),
        )
        for item in data.get("assertions", [])
    )

    return Scenario(
        name=str(data["name"]),
        profile=str(data["profile"]),
        scene=scene,
        commands=commands,
        assertions=assertions,
        steps=int(data["steps"]),
        dt=float(data["dt"]),
    )


def _readings(raw: object, default: tuple[float, ...]) -> tuple[float, ...]:
    """Normalize a raw TOML readings array into a tuple of floats, or fall back to a default.

    Args:
        raw: The value read from scene.thermal_readings / scene.power_readings (a TOML array,
            or None when the key is absent).
        default: The nominal-singleton fallback to use when raw is None.

    Returns:
        A tuple of floats: each element of raw coerced to float, or default when raw is None.

    Raises:
        TypeError: if raw is present but not an iterable of numeric values.
    """
    if raw is None:
        return default
    if not isinstance(raw, list):
        raise TypeError(f"scene readings must be a list, got {type(raw).__name__}")
    return tuple(float(value) for value in raw)


def _parse_tag(raw: object) -> AssertionTag:
    """Validate a raw TOML tag string against the allowed assertion tags.

    Args:
        raw: The tag value read from the TOML assertion table.

    Returns:
        The validated AssertionTag literal.

    Raises:
        ValueError: if the tag is not "frame-portable" or "realtime-only".
    """
    if raw == "frame-portable":
        return "frame-portable"
    if raw == "realtime-only":
        return "realtime-only"
    raise ValueError(f"unknown assertion tag: {raw!r}")
