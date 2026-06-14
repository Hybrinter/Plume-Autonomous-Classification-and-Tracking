"""Deterministic scenario builders + runner for the SIL telemetry recorder.

A ScenarioSpec is a fully declarative SIL run: how many steps to drive, the scene + sensor reading
scripts, the launch-lock state, any signed inbound telecommand packets, and two kinds of timed
hooks the recorder applies just before each step's cycle -- Injections (publish a prepared bus
message) and Actions (run a read/observe-or-stage callable on the system, e.g. stage a model blob
or flip the link state). Driving state changes through prepared bus messages / public driver calls
(never a flight code change) is exactly how the existing SIL tests steer the system; the recorder
captures the response passively.

The suite covers the nominal track plus every required fault/behavior path: thermal and power
over-limit -> SAFE -> stow, gimbal runaway, watchdog/process-died, EXIT_SAFE recovery via the
ARM/EXECUTE command path, hazardous ARM/EXECUTE gating, the launch-lock interlock, the model
upload -> activate -> rollback lifecycle, storage eviction, and downlink AOS/budget backpressure.
Faults that the deterministic ``step_once`` cannot raise organically (a gimbal encoder runaway, or
a watchdog miss when ``step_once`` synthesizes every app's heartbeat each step) are injected as the
FDIR input FaultEventMsg, which is documented per scenario.

Contains:
  - Injection / Action / ScenarioSpec / ScenarioRun: the declarative run + its captured result.
  - build_system / run_scenario: wire a SilSystem for a spec and capture a run.
  - SCENARIOS / scenario / scenario_names: the built-in scenario registry.
  - load_scenario_spec: adapt an existing scenarios/*.toml file into a ScenarioSpec.

Satisfies: REQ-OBS-SIL-001.
"""

from __future__ import annotations

# stdlib
import hashlib
import json
import tomllib
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

# internal
from flight.libs.commands import build_tc_packet, lookup_command
from flight.libs.config import PactConfig
from flight.libs.messages import (
    CommandMsg,
    FaultEventMsg,
    ModelStagedMsg,
)
from flight.libs.time import ManualClock
from flight.libs.types import DownlinkPriority, FaultCode, LinkState, MessageType, Ok
from sim.scene import build_frames, plume_detector
from sim.sil import SilSystem, build_sil_system

from tools.analysis.recorder import CaptureResult, PreStepHook, record_run

DEFAULT_UPLINK_KEY = b"sil-test-key-0000000000000000000"
_TS = "2026-06-01T00:00:00.000Z"
_TC_APID = 1

SystemAction = Callable[[SilSystem], None]


@dataclass(frozen=True, slots=True)
class Injection:
    """Publish a prepared bus message just before a given step's cycle.

    Fields:
        at_step: 1-based step index at which the message is published (in pre_step).
        message: the prepared, frozen bus message to publish.
    """

    at_step: int
    message: object


@dataclass(frozen=True, slots=True)
class Action:
    """Run a callable on the wired system just before a given step's cycle.

    Fields:
        at_step: 1-based step index at which the action runs (in pre_step).
        apply: the callable applied to the SilSystem (e.g. stage a model, flip the link state).
    """

    at_step: int
    apply: SystemAction


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """A fully declarative SIL run the recorder can drive and capture.

    Fields:
        name: stable scenario id (also the run's output directory stem).
        title: short human-readable title.
        description: one-paragraph description of what the run exercises (and any injection note).
        category: coarse grouping ("nominal", "thermal", "command", ...).
        steps: number of deterministic steps to run.
        dt: seconds advanced per step.
        num_frames: number of plume frames to render (>= steps keeps the payload active).
        seed: deterministic scene seed.
        thermal_readings: per-step thermal-sensor script (degC; holds last once exhausted).
        power_readings: per-step power-sensor script (W; holds last once exhausted).
        launch_lock_engaged: whether the launch lock starts ENGAGED.
        inbound_packets: signed CCSDS telecommand packets the station link delivers.
        injections: timed bus-message injections.
        actions: timed system actions.
        uplink_key: HMAC key the iss_iface app authenticates inbound packets with.
        config: the PactConfig to wire (scenarios may shrink quotas/budgets to exercise limits).
    """

    name: str
    title: str
    description: str
    category: str
    steps: int
    dt: float = 1.0
    num_frames: int = 0
    seed: int = 0
    thermal_readings: tuple[float, ...] = (25.0,)
    power_readings: tuple[float, ...] = (30.0,)
    launch_lock_engaged: bool = False
    inbound_packets: tuple[bytes, ...] = ()
    injections: tuple[Injection, ...] = ()
    actions: tuple[Action, ...] = ()
    uplink_key: bytes = DEFAULT_UPLINK_KEY
    config: PactConfig = field(default_factory=PactConfig)

    def frame_count(self) -> int:
        """Return the number of frames to render (num_frames, or steps when unset)."""
        return self.num_frames if self.num_frames > 0 else self.steps


@dataclass(frozen=True, slots=True)
class ScenarioRun:
    """A captured scenario run: the spec plus the recorder's CaptureResult."""

    spec: ScenarioSpec
    capture: CaptureResult


def build_system(spec: ScenarioSpec) -> SilSystem:
    """Wire a fresh SilSystem for a scenario spec (sim drivers, fresh bus + ManualClock)."""
    return build_sil_system(
        spec.config,
        ManualClock(),
        build_frames(spec.frame_count(), spec.seed),
        plume_detector(),
        inbound_packets=list(spec.inbound_packets),
        thermal_readings=list(spec.thermal_readings),
        power_readings=list(spec.power_readings),
        uplink_key=spec.uplink_key,
        launch_lock_engaged=spec.launch_lock_engaged,
    )


def _make_pre_step(spec: ScenarioSpec) -> PreStepHook:
    """Build the per-step hook that applies a spec's actions then injections at each step."""
    actions_by_step: dict[int, list[SystemAction]] = defaultdict(list)
    for action in spec.actions:
        actions_by_step[action.at_step].append(action.apply)
    messages_by_step: dict[int, list[object]] = defaultdict(list)
    for injection in spec.injections:
        messages_by_step[injection.at_step].append(injection.message)

    def pre_step(system: SilSystem, step: int) -> None:
        for apply in actions_by_step.get(step, ()):
            apply(system)
        for message in messages_by_step.get(step, ()):
            system.bus.publish(message)

    return pre_step


def run_scenario(spec: ScenarioSpec) -> ScenarioRun:
    """Wire and capture one scenario run end-to-end."""
    system = build_system(spec)
    return ScenarioRun(spec, record_run(system, spec.steps, spec.dt, _make_pre_step(spec)))


# ---------------------------------------------------------------------------
# Message + action builders for the built-in scenarios
# ---------------------------------------------------------------------------


def _command(
    command_id: str,
    target: str,
    params: dict[str, str | int | float | bool],
    source: str = "ground",
    seq: int = 1,
) -> CommandMsg:
    """Build a post-ingress CommandMsg (the command router routes by its stamped target)."""
    return CommandMsg(
        msg_type=MessageType.COMMAND,
        timestamp_utc=_TS,
        target=target,
        command_id=command_id,
        params=params,
        source=source,
        seq=seq,
    )


def _fault(code: FaultCode, subsystem: str, detail: str) -> FaultEventMsg:
    """Build a FaultEventMsg (the FDIR input the fault app routes through its SAFE policy)."""
    return FaultEventMsg(
        msg_type=MessageType.FAULT_EVENT,
        timestamp_utc=_TS,
        fault_code=code,
        subsystem=subsystem,
        detail=detail,
    )


def _stage_model_action(
    version: str, input_shape: tuple[int, ...], output_shape: tuple[int, ...]
) -> SystemAction:
    """Build an action that stores a model manifest blob and announces it via ModelStagedMsg.

    Faithfully drives the model-deploy stage path: it persists the manifest through the public
    StorageWriter and publishes the ModelStagedMsg (the message iss_iface emits after reassembly),
    so model_deploy validates the digest + manifest exactly as in flight.
    """
    blob = json.dumps(
        {"version": version, "input_shape": list(input_shape), "output_shape": list(output_shape)},
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()

    def apply(system: SilSystem) -> None:
        stored = system.apps.storage.store(
            f"model_{version}", blob, DownlinkPriority.SCIENCE_PRODUCT
        )
        if isinstance(stored, Ok):
            system.bus.publish(
                ModelStagedMsg(
                    msg_type=MessageType.MODEL_STAGED,
                    timestamp_utc=_TS,
                    entry_id=stored.value,
                    sha256=digest,
                    version=version,
                )
            )

    return apply


def _store_blob_action(item_id: str, nbytes: int) -> SystemAction:
    """Build an action that stores an opaque blob (to exercise the storage quota + eviction)."""
    blob = b"x" * nbytes

    def apply(system: SilSystem) -> None:
        system.apps.storage.store(item_id, blob, DownlinkPriority.SCIENCE_PRODUCT)

    return apply


def _set_link_action(state: LinkState) -> SystemAction:
    """Build an action that flips the sim station link AOS/LOS state."""

    def apply(system: SilSystem) -> None:
        system.station.set_link_state(state)

    return apply


def _storage_quota_config(max_bytes: int) -> PactConfig:
    """A PactConfig with the storage quota shrunk so eviction is reachable in a short run."""
    base = PactConfig()
    return replace(base, storage=replace(base.storage, max_storage_bytes=max_bytes))


def _downlink_budget_config(bytes_per_pass: int) -> PactConfig:
    """A PactConfig with the per-pass downlink byte budget shrunk so it binds in a short run."""
    base = PactConfig()
    return replace(base, comms=replace(base.comms, downlink_max_bytes_per_pass=bytes_per_pass))


# ---------------------------------------------------------------------------
# The built-in scenario suite
# ---------------------------------------------------------------------------


def _build_scenarios() -> dict[str, ScenarioSpec]:
    """Construct every built-in ScenarioSpec, keyed by name."""
    specs: list[ScenarioSpec] = [
        ScenarioSpec(
            name="nominal_tracking",
            title="Nominal plume tracking",
            description=(
                "Detect the scripted plume, transition IDLE -> ACQUIRING -> TRACKING, and slew the "
                "gimbal toward the target (azimuth positive, elevation negative). No faults; the "
                "system stays nominal the whole run."
            ),
            category="nominal",
            steps=14,
            num_frames=14,
        ),
        ScenarioSpec(
            name="thermal_over_limit_safe",
            title="Thermal over-limit -> SAFE -> stow",
            description=(
                "A thermal spike above the 80 C limit self-reports THERMAL_OVER_LIMIT; "
                "FDIR latches SAFE, the arbiter commands STOW, and the gimbal reaches the "
                "stow pose (stow switch engaged)."
            ),
            category="thermal",
            steps=16,
            num_frames=16,
            thermal_readings=(25.0, 25.0, 95.0),
        ),
        ScenarioSpec(
            name="power_over_limit_safe",
            title="Power over-limit -> SAFE",
            description=(
                "A power draw above the 55 W limit self-reports POWER_OVER_LIMIT; FDIR "
                "latches SAFE and the arbiter safes the gimbal."
            ),
            category="power",
            steps=12,
            num_frames=12,
            power_readings=(30.0, 30.0, 80.0),
        ),
        ScenarioSpec(
            name="gimbal_runaway",
            title="Gimbal runaway -> SAFE",
            description=(
                "Models an encoder runaway: the deterministic sim gimbal tracks commands "
                "faithfully, so a GIMBAL_RUNAWAY FaultEventMsg (the FDIR input the "
                "controller's runaway monitor would raise) is injected at step 3; FDIR "
                "routes it to SAFE."
            ),
            category="gimbal",
            steps=12,
            num_frames=12,
            injections=(
                Injection(
                    3, _fault(FaultCode.GIMBAL_RUNAWAY, "payload", "encoder rate divergence")
                ),
            ),
        ),
        ScenarioSpec(
            name="watchdog_process_died",
            title="Watchdog expiry / process died -> SAFE",
            description=(
                "step_once synthesizes every app's heartbeat each cycle, so a genuine miss "
                "is not reachable in the deterministic harness; a WATCHDOG_EXPIRE "
                "FaultEventMsg (the FDIR input a missed heartbeat would raise) is injected "
                "at step 3 and routed to SAFE. The per-subsystem watchdog miss/heartbeat-age "
                "signals stay nominal (synthesized)."
            ),
            category="watchdog",
            steps=12,
            num_frames=12,
            injections=(
                Injection(3, _fault(FaultCode.WATCHDOG_EXPIRE, "payload", "missed heartbeats")),
            ),
        ),
        ScenarioSpec(
            name="exit_safe_recovery",
            title="EXIT_SAFE recovery via ARM/EXECUTE",
            description=(
                "A thermal spike latches SAFE, then the spike clears; a ground EXIT_SAFE is "
                "ARMed (step 8) and EXECUTEd (step 9) through the command router and fault "
                "app, un-latching SAFE so the arbiter returns to operations and re-acquires "
                "the plume."
            ),
            category="recovery",
            steps=14,
            num_frames=14,
            thermal_readings=(25.0, 25.0, 95.0, 95.0, 25.0),
            injections=(
                Injection(8, _command("EXIT_SAFE", "fault", {"phase": "ARM"}, seq=1)),
                Injection(9, _command("EXIT_SAFE", "fault", {"phase": "EXECUTE"}, seq=2)),
            ),
        ),
        ScenarioSpec(
            name="arm_execute_command",
            title="Hazardous ARM/EXECUTE gating",
            description=(
                "Exercises the command router's two-step hazardous gate with "
                "RELEASE_LAUNCH_LOCK: an EXECUTE without a prior ARM is rejected (step 3), "
                "an ARM is accepted (step 5), then the matching EXECUTE routes to the "
                "mechanical app and releases the lock (step 6)."
            ),
            category="command",
            steps=10,
            num_frames=10,
            launch_lock_engaged=True,
            injections=(
                Injection(
                    3, _command("RELEASE_LAUNCH_LOCK", "mechanical", {"phase": "EXECUTE"}, seq=1)
                ),
                Injection(
                    5, _command("RELEASE_LAUNCH_LOCK", "mechanical", {"phase": "ARM"}, seq=2)
                ),
                Injection(
                    6, _command("RELEASE_LAUNCH_LOCK", "mechanical", {"phase": "EXECUTE"}, seq=3)
                ),
            ),
        ),
        ScenarioSpec(
            name="launch_lock_interlock",
            title="Launch-lock motion interlock",
            description=(
                "The launch lock starts ENGAGED; the mechanical app publishes the ENGAGED "
                "state and the payload inhibits gimbal motion (motion_inhibited) while still "
                "tracking the FSM, so the gimbal stays parked through the run -- the lock -> "
                "payload interlock direction."
            ),
            category="mechanical",
            steps=12,
            num_frames=12,
            launch_lock_engaged=True,
        ),
        ScenarioSpec(
            name="model_lifecycle",
            title="Model upload -> activate -> rollback",
            description=(
                "Stages a contract-valid model (step 2) and activates it (step 3) so it "
                "becomes ACTIVE with the factory model retained for rollback; then stages a "
                "contract-invalid model (step 5) and activates it (step 6), which fails the "
                "I/O-contract sanity check and auto-rolls-back (ROLLBACK_AVAILABLE) raising "
                "MODEL_CORRUPT (which also latches SAFE)."
            ),
            category="model",
            steps=12,
            num_frames=12,
            actions=(
                Action(2, _stage_model_action("v2", (1, 4, 256, 256), (1, 1, 256, 256))),
                Action(5, _stage_model_action("v3_bad", (1, 3, 128, 128), (1, 1, 128, 128))),
            ),
            injections=(
                Injection(3, _command("ACTIVATE_MODEL", "model_deploy", {"version": "v2"}, seq=1)),
                Injection(
                    6, _command("ACTIVATE_MODEL", "model_deploy", {"version": "v3_bad"}, seq=2)
                ),
            ),
        ),
        ScenarioSpec(
            name="storage_eviction",
            title="Storage quota eviction",
            description=(
                "With the storage quota shrunk to 8 KiB, opaque 3 KiB blobs are stored each "
                "step; once the quota is exceeded the storage service evicts the oldest "
                "entries (dropped_count rises) to keep the live byte total within the quota."
            ),
            category="storage",
            steps=12,
            num_frames=12,
            config=_storage_quota_config(8192),
            actions=tuple(
                Action(step, _store_blob_action(f"blob_{step}", 3000)) for step in range(2, 9)
            ),
        ),
        ScenarioSpec(
            name="downlink_aos_budget",
            title="Downlink AOS gate + per-pass budget",
            description=(
                "With the per-pass downlink budget shrunk to 256 bytes, the link drops to "
                "LOS (step 2) so the downlink queue backs up, then returns to AOS (step 8); "
                "draining is throttled by the per-pass budget, so the backlog clears only "
                "gradually."
            ),
            category="downlink",
            steps=16,
            num_frames=16,
            config=_downlink_budget_config(256),
            actions=(
                Action(2, _set_link_action(LinkState.LOS)),
                Action(8, _set_link_action(LinkState.AOS)),
            ),
        ),
        ScenarioSpec(
            name="command_ingress_auth",
            title="Signed command ingress -> route -> ack",
            description=(
                "A signed SET_THERMAL_LIMIT telecommand is delivered on the uplink; "
                "iss_iface authenticates and republishes it as a CommandMsg, the router "
                "dispatches it to the thermal app, and an ACCEPTED ack flows back -- the "
                "full ingress -> route -> execute command path."
            ),
            category="command",
            steps=8,
            num_frames=8,
            inbound_packets=(
                build_tc_packet(
                    "SET_THERMAL_LIMIT",
                    {"limit_c": 70.0},
                    "ground",
                    1,
                    DEFAULT_UPLINK_KEY,
                    _TC_APID,
                ),
            ),
        ),
    ]
    return {spec.name: spec for spec in specs}


SCENARIOS: dict[str, ScenarioSpec] = _build_scenarios()


def scenario(name: str) -> ScenarioSpec:
    """Return the built-in ScenarioSpec with the given name.

    Raises:
        KeyError: if name is not a built-in scenario.
    """
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario: {name!r} (have {sorted(SCENARIOS)})")
    return SCENARIOS[name]


def scenario_names() -> tuple[str, ...]:
    """Return every built-in scenario name, in declaration order."""
    return tuple(SCENARIOS)


def load_scenario_spec(path: str | Path) -> ScenarioSpec:
    """Adapt an existing scenarios/*.toml file (the GSE schema) into a ScenarioSpec.

    Args:
        path: filesystem path to a scenario TOML (name/profile/steps/dt + a [scene] table and
            optional [[commands]] tables).

    Returns:
        A ScenarioSpec that renders the same scene and injects each declared command as a
        post-ingress CommandMsg at its frame (the GSE assertion machinery is ignored -- capture
        is passive).

    Raises:
        OSError: if the file cannot be read.
        tomllib.TOMLDecodeError: if the file is not valid TOML.
        KeyError: if a required field is missing.

    Notes:
        Command targets are resolved from the flight command dictionary, so a declared command
        routes to the same subsystem it would in flight. This raises (test/CLI tooling) rather
        than returning a Result.
    """
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    scene = data["scene"]
    injections = tuple(
        Injection(
            int(cmd["at_frame"]),
            _command(
                str(cmd["command_id"]),
                _target_for(str(cmd["command_id"])),
                dict(cmd.get("params", {})),
                source=str(cmd["source"]),
                seq=int(cmd["seq"]),
            ),
        )
        for cmd in data.get("commands", [])
    )
    return ScenarioSpec(
        name=f"file_{data['name']}",
        title=f"Scenario file: {data['name']}",
        description=(
            f"Captured from scenarios/{data['name']}.toml (profile {data['profile']}); GSE "
            "assertions are not scored -- the run is captured passively for analysis."
        ),
        category="scenario-file",
        steps=int(data["steps"]),
        dt=float(data["dt"]),
        num_frames=int(scene["num_frames"]),
        seed=int(scene["seed"]),
        thermal_readings=_readings(scene.get("thermal_readings"), (25.0,)),
        power_readings=_readings(scene.get("power_readings"), (30.0,)),
        injections=injections,
    )


def _readings(raw: object, default: tuple[float, ...]) -> tuple[float, ...]:
    """Normalize an optional TOML readings array into a float tuple, or fall back to a default."""
    if raw is None:
        return default
    if not isinstance(raw, list):
        raise TypeError(f"scene readings must be a list, got {type(raw).__name__}")
    return tuple(float(value) for value in raw)


def _target_for(command_id: str) -> str:
    """Resolve a command_id to its canonical target subsystem via the flight command dictionary."""
    spec = lookup_command(command_id)
    return spec.value.target if isinstance(spec, Ok) else "core"
