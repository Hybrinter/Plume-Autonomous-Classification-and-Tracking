"""Payload application shell: HAL, capture pipeline, activation-driven graphs.

The shell owns no mode authority: SystemModeActivatedMsg snapshots select the
live payload graph through the pure accept_activation + runtime dispatch, the
mode-free ServoController executes the committed ControlReference, and the
capture loop only enqueues context-stamped CapturedVision records. One control
owner runs activation drain, safety evidence, routed commands, outer graph
cadence, servo execution, heartbeat, and every gimbal HAL call; the capture
loop may block on acquire/detect without touching graph or servo state.

Satisfies: REQ-AIML-COMP-001, REQ-AIML-COMP-002, REQ-OPER-HIGH-002.
"""

from __future__ import annotations

# stdlib
import math
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field, replace

# third-party
import numpy as np

# internal
from flight.hal.interfaces import (
    GimbalActuator,
    GimbalHealth,
    GimbalPosition,
    GimbalRateActuator,
    GimbalRateCommand,
    ImagingSensor,
    IssEphemeris,
    StorageWriter,
)
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import (
    FaultConfig,
    InferenceConfig,
    PactConfig,
    PreprocessingConfig,
    SensorConfig,
)
from flight.libs.messages import (
    CommandAckMsg,
    FaultEventMsg,
    GimbalCommandMsg,
    HeartbeatMsg,
    InferenceResultMsg,
    ProcessedFrameMsg,
    ProductRefMsg,
    RoutedCommandMsg,
    SafetyStateMsg,
    SystemModeActivatedMsg,
    SystemModeRequestMsg,
    SystemModeSyncRequestMsg,
    TelemetryEventMsg,
)
from flight.libs.time import Clock
from flight.libs.types import (
    AckStatus,
    ActivationKey,
    Band,
    DownlinkPriority,
    Err,
    FaultCode,
    FrameUsabilityTag,
    GimbalCommandMode,
    MessageType,
    MosaicFrame,
    Ok,
    Result,
    SystemMode,
)
from flight.payload.control import InnerTick, ServoController, ServoState
from flight.payload.gimbal.footprint import nominal_iss_state, tile_gsd_grid
from flight.payload.gimbal.integrity import check_integrity
from flight.payload.gimbal.intersect import CameraGeometry
from flight.payload.gimbal.request import (
    ControlReference,
    InhibitReference,
    PoseReference,
    StowReference,
)
from flight.payload.graphs import idle, init, operate, runtime, safe, stow
from flight.payload.graphs.base import (
    ActivationDisposition,
    ActivationSnapshot,
    ActivationState,
    EffectivePolicy,
    GraphId,
    NodeOutcome,
    SystemRequestIntent,
    TickInputs,
    accept_activation,
    validate_policy,
)
from flight.payload.graphs.operate.state import accept_vision
from flight.payload.graphs.parameters import GraphParameters
from flight.payload.imaging import CaptureDecision, CaptureSchedule, plan_capture, record_capture
from flight.payload.inference import DetectorBackend
from flight.payload.preprocess import (
    MosaicCalibration,
    SmearRateSource,
    calibrate_mosaic,
    compute_quality_flags,
    normalize_dn,
    select_bands,
    stack_channels,
)
from flight.payload.records import (
    CaptureContext,
    CapturedVision,
    HealthSample,
    IssSample,
    VisionSample,
)
from flight.payload.state import (
    PayloadState,
    graph_name_of,
    node_name_of,
    unactivated_reference,
)
from flight.payload.tracking import EncoderSample

_MODE_TO_GRAPH: dict[SystemMode, GraphId] = {
    SystemMode.IDLE: GraphId.IDLE,
    SystemMode.STOW: GraphId.STOW,
    SystemMode.SAFE: GraphId.SAFE,
    SystemMode.INIT: GraphId.INIT,
    SystemMode.OPERATE: GraphId.OPERATE,
}

_COMMAND_DEDUP_CAPACITY = 1024

_NodeOutcomeUnion = (
    NodeOutcome[idle.IdleNode]
    | NodeOutcome[safe.SafeNode]
    | NodeOutcome[stow.StowNode]
    | NodeOutcome[init.InitNode]
    | NodeOutcome[operate.OperateNode]
)


@dataclass(frozen=True, slots=True)
class TickOutcome:
    """Summary of one payload cycle, returned for telemetry/testing.

    Attributes:
        frame_id: The frame_id of the processed raw frame (0 for loop ticks).
        fault: FaultCode if preprocessing or detection failed this frame, else None.
        command_issued: True if a GimbalCommandMsg was published this cycle.
        payload_graph: GraphId value of the active graph, "" when unactivated.
        payload_node: Node value within the active graph, "" when unactivated.
    """

    frame_id: int
    fault: FaultCode | None
    command_issued: bool
    payload_graph: str = ""
    payload_node: str = ""


@dataclass(frozen=True, slots=True)
class ReferenceCommit:
    """Result of one control-reference commit attempt.

    Attributes:
        state: Payload state carrying the recorded reference.
        command_issued: True when a GimbalCommandMsg audit was published.
        fault: The HAL fault code when the reference metadata operation
            failed, else None; containment is latched on any fault.
    """

    state: PayloadState
    command_issued: bool
    fault: FaultCode | None


@dataclass(slots=True)
class ContainmentState:
    """Local motion containment: hardware inhibit latch plus fault evidence.

    local_latched never clears on healthy evidence or an ordinary activation;
    only the authorized-recovery contract releases it. latest_safety is the
    most recent fault-owned SafetyStateMsg; consumed_recovery_ids holds the
    request ids already spent on a release so a replayed record cannot
    re-release, and last_evidence_sequence tracks the newest consumed
    fault evidence sequence for the strictly-increasing check. generation
    increments on every latch engagement so in-flight capture work carrying a
    stale safety token is dropped.
    """

    local_latched: bool = False
    latest_safety: SafetyStateMsg | None = None
    consumed_recovery_ids: set[str] = field(default_factory=set)
    last_evidence_sequence: int = 0
    inhibit_applied: bool = False
    generation: int = 0


@dataclass(slots=True)
class ActuatorSafety:
    """Mutable shell state for driver evidence and bounded recovery accounting."""

    last_feedback_s: float | None = None
    recovery_window_start_s: float | None = None
    recovery_attempts: int = 0
    recovery_pending_reset: bool = False
    last_health_publish_s: float | None = None
    last_health: GimbalHealth | None = None


@dataclass(slots=True)
class EncoderStream:
    """Timestamped encoder samples shared by frame association and control."""

    samples: deque[EncoderSample] = field(default_factory=lambda: deque(maxlen=4096))
    consumed_ids: set[str] = field(default_factory=set)


@dataclass(slots=True)
class CommandDedup:
    """Bounded (source, seq, command_id) tracker; duplicates never reexecute."""

    order: deque[tuple[str, int, str]] = field(default_factory=deque)
    index: set[tuple[str, int, str]] = field(default_factory=set)

    def seen_or_add(self, key: tuple[str, int, str]) -> bool:
        """Return True when the key was already seen; record it otherwise."""
        if key in self.index:
            return True
        self.index.add(key)
        self.order.append(key)
        while len(self.order) > _COMMAND_DEDUP_CAPACITY:
            self.index.discard(self.order.popleft())
        return False


@dataclass(slots=True)
class StowProgress:
    """Shell bookkeeping for the committed StowReference drive cycle."""

    armed: bool = False
    complete: bool = False


@dataclass(slots=True)
class CaptureShell:
    """Mutable capture-loop bookkeeping: transport state plus planning phase.

    acquisition_on and applied_capture mirror the driver transport; schedule
    is the pure plan/record phase state, reset whenever the capture context
    changes. pending_fault queues an imaging fault for the control owner to
    consume and contain; applied_revision marks the last policy revision that
    emitted the compact applied-policy telemetry record.
    """

    acquisition_on: bool = False
    applied_capture: tuple[float, float] | None = None
    schedule: CaptureSchedule = field(default_factory=CaptureSchedule)
    pending_fault: FaultCode | None = None
    applied_revision: int | None = None


@dataclass(slots=True)
class RuntimeShell:
    """Latest control-committed PayloadState shared with the capture loop.

    The control owner installs the newest immutable state under state_lock on
    every control-method entry and exit; the capture loop reads it for token
    comparison but never installs. generation mirrors control_revision so a
    superseded computed actuation or a stale capture context is detectable.
    """

    state: PayloadState | None = None
    generation: int = 0


@dataclass(frozen=True)
class PayloadApp:
    """Payload subsystem app: imperative shell around the graph runtime.

    Attributes:
        sensor: ImagingSensor driver.
        gimbal: GimbalActuator driver.
        ephemeris: IssEphemeris driver.
        detector: DetectorBackend.
        servo: Mode-free ServoController (rate/position law + inner PI).
        params: GraphParameters projection for pure graph functions.
        bus: Typed MessageBus.
        clock: Injected Clock.
        calib: MosaicCalibration.
        storage: StorageWriter.
        activation_epoch: Composition-root epoch; the app never generates one.
        activation_sub, safety_sub, cmd_sub: Bus subscriptions.
        containment: Local inhibit latch + fault-owned safety evidence.
        vision_queue: In-process CapturedVision queue (not the MessageBus).
        state_lock: Short lock for immutable snapshot install only.
    """

    sensor: ImagingSensor
    gimbal: GimbalActuator
    ephemeris: IssEphemeris
    detector: DetectorBackend
    servo: ServoController
    params: GraphParameters
    bus: MessageBus
    clock: Clock
    calib: MosaicCalibration
    storage: StorageWriter
    sensor_cfg: SensorConfig
    inference_cfg: InferenceConfig
    preprocessing_cfg: PreprocessingConfig
    fault_cfg: FaultConfig
    activation_epoch: str
    activation_sub: Subscription[SystemModeActivatedMsg]
    safety_sub: Subscription[SafetyStateMsg]
    local_fault_sub: Subscription[FaultEventMsg]
    cmd_sub: Subscription[RoutedCommandMsg]
    containment: ContainmentState = field(default_factory=ContainmentState)
    vision_queue: deque[CapturedVision] = field(default_factory=lambda: deque(maxlen=4))
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    actuator_io_lock: threading.Lock = field(default_factory=threading.Lock)
    actuator_safety: ActuatorSafety = field(default_factory=ActuatorSafety)
    encoder_stream: EncoderStream = field(default_factory=EncoderStream)
    command_dedup: CommandDedup = field(default_factory=CommandDedup)
    stow_progress: StowProgress = field(default_factory=StowProgress)
    emitted_intents: set[tuple[int, str]] = field(default_factory=set)
    capture_shell: CaptureShell = field(default_factory=CaptureShell)
    runtime_shell: RuntimeShell = field(default_factory=RuntimeShell)

    @staticmethod
    def from_config(
        cfg: PactConfig,
        sensor: ImagingSensor,
        gimbal: GimbalActuator,
        ephemeris: IssEphemeris,
        detector: DetectorBackend,
        bus: MessageBus,
        clock: Clock,
        calib: MosaicCalibration,
        storage: StorageWriter,
        activation_epoch: str,
    ) -> PayloadApp:
        """Assemble a PayloadApp from a PactConfig and injected services.

        Raises:
            ValueError: Invalid channel layout, inference geometry, or an
                operate/node payload policy that resolves to an invalid
                combination; an invalid policy is a startup configuration
                failure, never a degraded runtime one.
        """
        if (
            cfg.sensor.height_px != cfg.inference.input_height_px
            or cfg.sensor.width_px != cfg.inference.input_width_px
        ):
            raise ValueError("sensor frame must equal the inference input size")
        if sorted(cfg.sensor.channel_layout) != sorted(b.value for b in Band):
            raise ValueError("channel_layout must name each Band exactly once")
        if any(b not in cfg.sensor.channel_layout for b in cfg.inference.input_bands):
            raise ValueError("input_bands must be a subset of channel_layout")
        detailed = not isinstance(gimbal, GimbalRateActuator)
        params = GraphParameters(config=cfg, detailed_plant=detailed)
        policy_cfg = cfg.payload_policy
        node_overrides = (
            policy_cfg.tracking,
            policy_cfg.rewind,
            policy_cfg.fast_rewind,
            policy_cfg.hold,
        )
        for override in (None, *node_overrides):
            resolved = params.operating_policy(override)
            if isinstance(resolved, Err):
                raise ValueError(f"invalid payload policy: {resolved.error.value}")
        return PayloadApp(
            sensor=sensor,
            gimbal=gimbal,
            ephemeris=ephemeris,
            detector=detector,
            servo=ServoController.from_config(cfg.controller, cfg.gimbal),
            params=params,
            bus=bus,
            clock=clock,
            calib=calib,
            storage=storage,
            sensor_cfg=cfg.sensor,
            inference_cfg=cfg.inference,
            preprocessing_cfg=cfg.preprocessing,
            fault_cfg=cfg.fault,
            activation_epoch=activation_epoch,
            activation_sub=bus.subscribe(SystemModeActivatedMsg),
            safety_sub=bus.subscribe(SafetyStateMsg),
            local_fault_sub=bus.subscribe(FaultEventMsg),
            cmd_sub=bus.subscribe(RoutedCommandMsg),
            vision_queue=deque(maxlen=cfg.controller.vision.queue_depth),
        )

    def initial_state(self) -> PayloadState:
        """Boot state: unactivated, inhibited, imaging off, no graph."""
        return PayloadState(
            servo=self.servo.initial_state(),
            activation=ActivationState(expected_epoch=self.activation_epoch),
            graph=None,
            reference=unactivated_reference(),
            policy=self.params.default_policy(False),
            last_outer_s=None,
        )

    def _params(self, state: PayloadState) -> GraphParameters:
        """Graph parameters stamped with the state's current policy revision."""
        return replace(self.params, policy_revision=state.policy_revision)

    def _install(self, state: PayloadState) -> None:
        """Install the newest committed state for capture readers (control only)."""
        with self.state_lock:
            self.runtime_shell.state = state
            self.runtime_shell.generation = state.control_revision

    def _latest(self, fallback: PayloadState) -> PayloadState:
        """Return the newest control-committed state, or the passed fallback."""
        with self.state_lock:
            shell = self.runtime_shell.state
        return shell if shell is not None else fallback

    def _encoder_snapshot(self) -> tuple[EncoderSample, ...]:
        """Locked tuple copy of the control-recorded encoder history."""
        with self.state_lock:
            return tuple(self.encoder_stream.samples)

    def poll_activations(self, state: PayloadState, now: float) -> PayloadState:
        """Drain safety evidence then activation records, in bus order.

        SAFE is never coalesced: each accepted SAFE activation inhibits the
        actuator immediately before any later activation in the drain is
        applied. Duplicates reenter nothing; stale snapshots are ignored; a
        sequence gap is accepted with telemetry; a conflicting or wrong-epoch
        snapshot raises a synchronization fault and latches local containment.
        """
        unsafe_this_drain = self._poll_safety(now) | self._poll_local_faults()
        current = state
        self._install(current)
        while not self.activation_sub.empty():
            msg = self.activation_sub.get_nowait()
            snapshot = ActivationSnapshot(
                key=ActivationKey(epoch=msg.epoch, sequence=msg.sequence),
                graph_id=_MODE_TO_GRAPH[msg.active_mode],
                previous_graph=(
                    _MODE_TO_GRAPH[msg.previous_mode] if msg.previous_mode is not None else None
                ),
                reason=msg.reason,
                request_id=msg.request_id,
                recovery_authorized=msg.recovery_authorized,
            )
            decision = accept_activation(current.activation, snapshot)
            if isinstance(decision, Err):
                self._publish_fault(FaultCode.GIMBAL_FAULT, "activation synchronization conflict")
                self._latch_containment("activation conflict")
                continue
            result = decision.value
            current = replace(current, activation=result.state)
            if result.disposition is not ActivationDisposition.ACCEPTED:
                continue
            if result.gap:
                self._publish_telemetry(
                    "activation_gap",
                    {
                        "epoch": msg.epoch,
                        "sequence": msg.sequence,
                    },
                )
            if snapshot.graph_id is GraphId.SAFE:
                self._latch_containment("safe_activation")
            current = self._reenter(current, snapshot, now)
        last = current.activation.last
        if (
            not unsafe_this_drain
            and self.containment.local_latched
            and last is not None
            and last.recovery_authorized
            and last.graph_id is GraphId.IDLE
            and last.previous_graph is GraphId.SAFE
            and last.request_id
        ):
            self._try_release_containment(last, now)
        self._install(current)
        return current

    def _poll_safety(self, now: float) -> bool:
        """Drain fault-owned SafetyStateMsg; a latched SAFE inhibits locally.

        Evidence qualifies only under the injected epoch with a nonnegative,
        strictly increasing sequence and a finite nonfuture observation time;
        stale or malformed records are ignored rather than overwriting the
        latest accepted evidence. Returns True when any drained record carried
        a SAFE latch or active faults, so a recovery release can never slip
        through in the same drain.
        """
        unsafe = False
        while not self.safety_sub.empty():
            evidence = self.safety_sub.get_nowait()
            if (
                evidence.evidence_epoch != self.activation_epoch
                or evidence.evidence_sequence < 0
                or not math.isfinite(evidence.observed_s)
                or evidence.observed_s > now
            ):
                continue
            if evidence.safe_latched or evidence.active_faults:
                unsafe = True
            latest = self.containment.latest_safety
            if latest is not None and evidence.evidence_sequence <= latest.evidence_sequence:
                continue
            self.containment.latest_safety = evidence
            if evidence.safe_latched:
                self._latch_containment("fault_safety_evidence")
        return unsafe

    def _poll_local_faults(self) -> bool:
        """Drain payload-origin FaultEventMsg; containing codes engage the latch.

        A payload-origin containing fault must contain the actuator at the
        next control poll rather than wait for the external fault app's tick
        to request SAFE. Only codes in _CONTAINING_FAULTS from the "payload"
        subsystem latch here; every drained containing record also marks the
        drain unsafe so a matching recovery clear in the same pass can never
        release. A queued imaging pending_fault from the capture loop is
        consumed once under the same drain and always latches, so camera
        faults contain through the control owner without the capture loop
        ever calling the gimbal.
        """
        batch: list[FaultEventMsg] = []
        while not self.local_fault_sub.empty():
            batch.append(self.local_fault_sub.get_nowait())
        with self.state_lock:
            pending = self.capture_shell.pending_fault
            self.capture_shell.pending_fault = None
        unsafe = (
            any(
                event.subsystem == "payload" and event.fault_code in _CONTAINING_FAULTS
                for event in batch
            )
            or pending is not None
        )
        if unsafe:
            self._latch_containment("payload local fault")
        return unsafe

    def _mark_containment_latched(self) -> None:
        """Atomically engage the local latch and bump its generation token."""
        with self.state_lock:
            if not self.containment.local_latched:
                self.containment.generation += 1
            self.containment.local_latched = True

    def _latch_containment(self, reason: str) -> None:
        """Engage the local inhibit latch and request driver containment once."""
        self._mark_containment_latched()
        if not self.containment.inhibit_applied:
            self.containment.inhibit_applied = self._inhibit_motion(reason)

    def _reenter(
        self, state: PayloadState, snapshot: ActivationSnapshot, now: float
    ) -> PayloadState:
        """Reenter initial graph state under a newly accepted activation.

        Flushes queued vision, resets the servo integrator/integrity (valid
        physical encoder history is retained), inhibits until the destination
        graph emits its first outcome, and bumps both revisions so queued
        capture and computed actuation from the old activation are obsolete.
        """
        with self.state_lock:
            self.vision_queue.clear()
        self._inhibit_motion("activation")
        self.stow_progress.armed = False
        self.stow_progress.complete = False
        self.emitted_intents.clear()
        servo = replace(
            state.servo,
            inner=replace(state.servo.inner, integrator=0.0, last_tau_nm=0.0),
            integrity=replace(state.servo.integrity, freeze_strikes=0),
            commanded_rate_rad_s=0.0,
        )
        params = self._params(state)
        entry = runtime.entry_policy(snapshot.graph_id, params)
        if isinstance(entry, Err):
            # Startup validation normally prevents this path: fail closed off
            # and queue control-owned containment rather than run an
            # unresolved policy.
            self._imaging_fault(entry.error, "entry policy resolution failed")
        policy = entry.value if isinstance(entry, Ok) else self.params.default_policy(False)
        reset = replace(
            state,
            servo=servo,
            graph=None,
            reference=InhibitReference("activation"),
            policy=policy,
            policy_revision=state.policy_revision + 1,
            control_revision=state.control_revision + 1,
            last_outer_s=now,
        )
        inputs = self._tick_inputs(reset, now, timestamp_utc=self.clock.wall_clock_iso())
        graph = runtime.initial_state(snapshot.graph_id, inputs, params)
        return replace(reset, graph=graph)

    def _try_release_containment(self, snapshot: ActivationSnapshot, now: float) -> None:
        """Release the local latch only under the authorized-recovery contract.

        Pending is not failure: while the current authorized IDLE activation's
        request_id has no matching fresh fault-owned release evidence yet, the
        latch simply stays engaged without a fault. Once evidence matches --
        same request_id, injected epoch, strictly increasing sequence, finite
        nonfuture observed_s inside the watchdog interval, no active faults or
        SAFE latch -- the authorization is consumed exactly once and hardware is
        validated: feedback_valid, inhibit_confirmed, finite last_feedback_s
        aged 0..feedback_max_age_s, and a finite, nonfuture, hardware-bounded
        encoder sample with nonnegative variance. A failed hardware check emits
        one fault and keeps the latch; the consumed request_id cannot replay.
        """
        request_id = snapshot.request_id
        if not request_id or request_id in self.containment.consumed_recovery_ids:
            return
        evidence = self.containment.latest_safety
        if (
            evidence is None
            or evidence.recovery_request_id != request_id
            or evidence.evidence_epoch != self.activation_epoch
            or evidence.evidence_sequence <= self.containment.last_evidence_sequence
            or not math.isfinite(evidence.observed_s)
            or not 0.0 <= now - evidence.observed_s <= self.fault_cfg.watchdog_interval_s
            or evidence.active_faults
            or evidence.safe_latched
        ):
            return
        self.containment.consumed_recovery_ids.add(request_id)
        self.containment.last_evidence_sequence = evidence.evidence_sequence
        health_result = self._read_health()
        health = health_result.value if isinstance(health_result, Ok) else None
        if not (health is not None and health.feedback_valid and health.inhibit_confirmed):
            self._publish_fault(FaultCode.GIMBAL_FAULT, "recovery release: unconfirmed actuator")
            return
        last_fb = health.last_feedback_s
        if (
            last_fb is None
            or not math.isfinite(last_fb)
            or last_fb - now > 1.0e-12
            or now - last_fb > self.params.feedback_max_age_s
        ):
            self._publish_fault(FaultCode.GIMBAL_FAULT, "recovery release: stale feedback")
            return
        if not self._encoder_healthy(now):
            self._publish_fault(FaultCode.GIMBAL_FAULT, "recovery release: invalid encoder")
            return
        with self.state_lock:
            self.containment.local_latched = False
            self.containment.inhibit_applied = False
        self.actuator_safety.recovery_pending_reset = True

    def _encoder_healthy(self, now: float) -> bool:
        """Latest encoder sample finite, fresh, nonfuture, and inside the hardware envelope."""
        if not self.encoder_stream.samples:
            return False
        sample = max(self.encoder_stream.samples, key=lambda s: s.t_s)
        if not (
            math.isfinite(sample.t_s)
            and math.isfinite(sample.angle_rad)
            and math.isfinite(sample.angle_variance_rad2)
        ):
            return False
        if sample.angle_variance_rad2 < 0.0:
            return False
        if sample.t_s - now > 1.0e-12 or now - sample.t_s > self.params.feedback_max_age_s:
            return False
        lo = math.radians(self.params.config.gimbal.el_hw_min_deg) - 1e-12
        hi = math.radians(self.params.config.gimbal.el_hw_max_deg) + 1e-12
        return lo <= sample.angle_rad <= hi

    def _read_health(self) -> Result[GimbalHealth, FaultCode]:
        """Read actuator health under the driver-I/O lock; retain the last value."""
        with self.actuator_io_lock:
            health = self.gimbal.read_health()
        if isinstance(health, Ok):
            self.actuator_safety.last_health = health.value
        return health

    def handle_commands(
        self, state: PayloadState, now: float, vision: CapturedVision | None = None
    ) -> PayloadState:
        """Drain routed commands onto the graph command path, one ack each.

        Deterministic/test seam: commits at most one directed graph edge under
        the current tick inputs. (source, seq, command_id) duplicates never
        reexecute and never emit a competing ack. A flagged vision sample blocks
        every command this pass; without an accepted activation, without an
        encoder, or while contained, each distinct command gets one correlated
        NACK. In run/SIL the same guard path is invoked per outer tick inside
        advance_outer, after the flagged-vision graph step.
        """
        current = state
        self._install(current)
        self._read_health()
        if vision is not None and self._context_stale(vision.context, current):
            vision = None
        inputs = (
            None
            if current.graph is None
            else self._tick_inputs(current, now, timestamp_utc="", vision=vision)
        )
        flagged = False
        if vision is not None and isinstance(current.graph, operate.State) and inputs is not None:
            accepted = accept_vision(current.graph, inputs, self._params(current))
            flagged = accepted is not None and accepted.sample.mode_flags != 0
        current, _edge = self._drain_commands(
            current,
            now,
            None if flagged else inputs,
            blocked_reason="flagged vision" if flagged else None,
        )
        self._install(current)
        return current

    def _drain_commands(
        self,
        state: PayloadState,
        now: float,
        inputs: TickInputs | None,
        *,
        blocked_reason: str | None = None,
    ) -> tuple[PayloadState, bool]:
        """Drain the routed-command queue once; stop after one accepted edge.

        Returns the updated state and True when a command committed a directed
        graph edge this pass (the caller must not also step the graph).
        """
        current = state
        edge = False
        while not self.cmd_sub.empty():
            command = self.cmd_sub.get_nowait()
            if command.target != "payload":
                continue
            key = (command.source, command.seq, command.command_id)
            if self.command_dedup.seen_or_add(key):
                continue
            if current.graph is None or current.activation.last is None:
                self._ack_command(
                    command, AckStatus.REJECTED, FaultCode.COMMAND_INVALID, "no active graph"
                )
                continue
            if self.containment.local_latched:
                self._ack_command(
                    command, AckStatus.REJECTED, FaultCode.COMMAND_INVALID, "inhibited"
                )
                continue
            if blocked_reason is not None:
                self._ack_command(
                    command, AckStatus.REJECTED, FaultCode.COMMAND_INVALID, blocked_reason
                )
                continue
            if inputs is None or inputs.encoder is None:
                self._ack_command(
                    command, AckStatus.REJECTED, FaultCode.COMMAND_INVALID, "no encoder"
                )
                continue
            applied = runtime.apply_command(current.graph, command, inputs, self._params(current))
            if isinstance(applied, Err):
                self._ack_command(command, AckStatus.REJECTED, applied.error, "command rejected")
                continue
            outcome = applied.value
            committed = replace(current, graph=outcome.state)
            committed, _issued = self._commit_outcome(committed, outcome.outcome.outcome, now)
            if self.containment.local_latched:
                disabled = self._params(current).default_policy(False)
                current = replace(
                    current,
                    reference=InhibitReference("actuator failure"),
                    policy=disabled,
                    policy_revision=current.policy_revision
                    + (1 if disabled != current.policy else 0),
                )
                self._ack_command(
                    command,
                    AckStatus.REJECTED,
                    FaultCode.GIMBAL_FAULT,
                    "actuator reference failed",
                )
                break
            current = committed
            self._ack_command(command, AckStatus.ACCEPTED, FaultCode.NONE, "executed")
            edge = True
            break  # at most one directed edge per outer tick
        return current, edge

    def _tick_inputs(
        self,
        state: PayloadState,
        now: float,
        *,
        timestamp_utc: str,
        encoder: EncoderSample | None = None,
        vision: CapturedVision | None = None,
        navigation: IssSample | None = None,
        stow_complete: bool = False,
    ) -> TickInputs:
        """Assemble the pure TickInputs for one evaluation."""
        last = state.activation.last
        key = last.key if last is not None else ActivationKey(epoch="", sequence=-1)
        if encoder is None and self.encoder_stream.samples:
            encoder = max(self.encoder_stream.samples, key=lambda s: s.t_s)
        health = self.actuator_safety.last_health
        return TickInputs(
            now_s=now,
            timestamp_utc=timestamp_utc,
            activation_key=key,
            encoder=encoder,
            navigation=navigation,
            vision=vision,
            health=HealthSample(
                feedback_valid=health is not None and health.feedback_valid,
                inhibit_confirmed=health is not None and health.inhibit_confirmed,
                contained=self.containment.local_latched,
            ),
            stow_complete=stow_complete,
        )

    def _commit_outcome(
        self,
        state: PayloadState,
        outcome: _NodeOutcomeUnion,
        now: float,
    ) -> tuple[PayloadState, bool]:
        """Commit a node outcome: publish events/faults, requests, reference.

        Graph faults and system requests are processed before any reference
        action; graph telemetry events are deferred until the reference commit
        succeeds, so a failed actuator metadata operation never publishes the
        committed-edge events for a transition that did not take effect.
        """
        for code in outcome.faults:
            self._publish_fault(code, "graph outcome fault")
            if code in _CONTAINING_FAULTS:
                self._latch_containment(code.value)
        intent = outcome.system_request
        if intent is not None:
            self._emit_intent(intent, state)
        reference = outcome.reference
        policy = outcome.policy
        current = state
        issued = False
        if policy != state.policy:
            current = replace(current, policy=policy, policy_revision=current.policy_revision + 1)
        if reference != state.reference:
            commit = self._commit_reference(current, reference, now)
            current = commit.state
            issued = commit.command_issued
            if commit.fault is not None:
                return current, issued
        last = state.activation.last
        for event in outcome.events:
            self.bus.publish(
                replace(
                    event,
                    payload={
                        **event.payload,
                        "activation_epoch": last.key.epoch if last is not None else "",
                        "activation_sequence": (last.key.sequence if last is not None else -1),
                    },
                )
            )
        return current, issued

    def _publish_pointing(
        self,
        state: PayloadState,
        now: float,
        encoder: EncoderSample,
        vision: CapturedVision | None,
    ) -> None:
        """Emit the compact pointing telemetry record after a committed tick.

        TRACKING reports the authoritative graph residual/CoG/rate terms;
        outside TRACKING the residual- and chrono-derived fields are None
        rather than fabricated, while encoder/reference identity stays
        meaningful for every graph.
        """
        graph = state.graph
        last = state.activation.last
        op = graph if isinstance(graph, operate.State) else None
        tracking = op is not None and op.node is operate.OperateNode.TRACKING
        tr = op if tracking else None
        decision = op.last_rate_decision if op is not None else None
        checkpoint = op.residual_history.checkpoint if op is not None else None
        anchor_angle = checkpoint.encoder_angle_rad if checkpoint is not None else None
        span_s = max(0.0, encoder.t_s - checkpoint.t_s) if checkpoint is not None else None
        residual_cfg = self.params.config.controller.residual
        self.bus.publish(
            TelemetryEventMsg(
                msg_type=MessageType.TELEMETRY_EVENT,
                timestamp_utc=self.clock.wall_clock_iso(),
                subsystem="payload",
                event_name="pointing",
                payload={
                    "payload_graph": graph_name_of(state),
                    "payload_node": node_name_of(state),
                    "activation_epoch": last.key.epoch if last is not None else "",
                    "activation_sequence": last.key.sequence if last is not None else -1,
                    "e": float(tr.residual.x[0]) if tr is not None else None,
                    "r": state.servo.commanded_rate_rad_s,
                    "tau": state.servo.inner.last_tau_nm,
                    "omega_t_nom": tr.target.last_omega_t_nom if tr is not None else None,
                    "omega_az": tr.target.last_omega_az_nom if tr is not None else None,
                    "omega_scene_el": (tr.target.last_omega_scene_el if tr is not None else None),
                    "omega_t_res": float(tr.residual.x[1]) if tr is not None else None,
                    "omega_t_total": (
                        tr.target.last_omega_t_nom + float(tr.residual.x[1])
                        if tr is not None
                        else None
                    ),
                    "requested_relative_rate_rad_s": (
                        decision.requested_relative_rate_rad_s if decision else None
                    ),
                    "hardware_limited": decision.hardware_limited if decision else None,
                    "science_limited": decision.science_limited if decision else None,
                    "rewind_elapsed_s": (
                        max(0.0, now - op.rewind_entered_s)
                        if op is not None and op.rewind_entered_s is not None
                        else 0.0
                    ),
                    "y_m": state.servo.encoder.measured_rate_rad_s,
                    "P00": float(tr.residual.P[0, 0]) if tr is not None else None,
                    "P01": float(tr.residual.P[0, 1]) if tr is not None else None,
                    "P10": float(tr.residual.P[1, 0]) if tr is not None else None,
                    "P11": float(tr.residual.P[1, 1]) if tr is not None else None,
                    "encoder_anchor_t_s": (checkpoint.t_s if checkpoint is not None else None),
                    "encoder_endpoint_id": encoder.sample_id,
                    "encoder_endpoint_t_s": encoder.t_s,
                    "encoder_net_displacement_rad": (
                        encoder.angle_rad - anchor_angle if anchor_angle is not None else None
                    ),
                    "encoder_sample_age_s": max(0.0, now - encoder.t_s),
                    "encoder_span_s": span_s,
                    "process_q_e_rad2": (
                        residual_cfg.q_accel_rad2_s3 * span_s**3 / 3.0
                        if span_s is not None
                        else None
                    ),
                    "process_q_rate_rad2_s2": (
                        residual_cfg.q_accel_rad2_s3 * span_s if span_s is not None else None
                    ),
                    "encoder_endpoint_covariance_rad2": (
                        checkpoint.encoder_endpoint_variance_rad2 + encoder.angle_variance_rad2
                        if checkpoint is not None
                        else None
                    ),
                    "reversal_covariance_per_event_rad2": (
                        residual_cfg.reversal_variance_rad2 if checkpoint is not None else None
                    ),
                    "vision_event_id": vision.sample.frame_id if vision is not None else "",
                    "vision_shutter_s": vision.sample.t_s if vision is not None else None,
                    "vision_arrival_s": now if vision is not None else None,
                    "vision_disposition": (
                        op.vision_disposition.value
                        if op is not None and op.vision_disposition is not None
                        else None
                    ),
                },
            )
        )

    def _emit_intent(self, intent: SystemRequestIntent, state: PayloadState) -> None:
        """Publish a deduped payload-owned SystemModeRequestMsg per activation."""
        last = state.activation.last
        seq = last.key.sequence if last is not None else -1
        if (seq, intent.value) in self.emitted_intents:
            return
        self.emitted_intents.add((seq, intent.value))
        mode = SystemMode.SAFE if intent is SystemRequestIntent.SAFE else SystemMode.IDLE
        self.bus.publish(
            SystemModeRequestMsg(
                msg_type=MessageType.SYSTEM_MODE_REQUEST,
                timestamp_utc=self.clock.wall_clock_iso(),
                request_id=str(uuid.uuid4()),
                requested_mode=mode,
                requested_by="payload",
                reason=f"graph_intent:{intent.value}",
                activation_key=last.key if last is not None else None,
            )
        )

    def _commit_reference(
        self, state: PayloadState, reference: ControlReference, now: float
    ) -> ReferenceCommit:
        """Commit a new control reference; arm stow/goto HAL surfaces as needed.

        While contained, or when the passed state is no longer the installed
        control revision (a newer activation superseded it), the reference is
        recorded but no stow/goto HAL metadata or audit command is produced.
        Any reference-metadata HAL failure latches containment immediately and
        returns fault=<code> with no successful audit.
        """
        current = replace(state, reference=reference)
        latest = self._latest(state)
        superseded = (
            latest.control_revision != state.control_revision
            or latest.activation.last != state.activation.last
        )
        if self.containment.local_latched or superseded:
            return ReferenceCommit(current, False, None)
        if isinstance(reference, StowReference):
            if not self.stow_progress.armed:
                with self.actuator_io_lock:
                    result = self.gimbal.stow()
                if isinstance(result, Err):
                    self._record_actuator_failure(result.error, now)
                    self._latch_containment("actuator reference failed")
                    self._publish_fault(result.error, "gimbal stow arm failed")
                    return ReferenceCommit(current, False, result.error)
                self.stow_progress.armed = True
            self.stow_progress.complete = False
            self._publish_gimbal_command(GimbalCommandMode.STOW, 0.0, current, "stow_reference")
            return ReferenceCommit(current, True, None)
        if isinstance(reference, PoseReference):
            if not self._rate_mode():
                with self.actuator_io_lock:
                    send = self.gimbal.goto_angle(math.degrees(reference.target_rad))
                if isinstance(send, Err):
                    self._record_actuator_failure(send.error, now)
                    self._latch_containment("actuator reference failed")
                    self._publish_fault(send.error, "gimbal pose latch failed")
                    return ReferenceCommit(current, False, send.error)
            self._publish_gimbal_command(
                GimbalCommandMode.ABSOLUTE,
                math.degrees(reference.target_rad),
                current,
                "pose_reference",
            )
            return ReferenceCommit(current, True, None)
        if isinstance(reference, InhibitReference):
            if not self._inhibit_motion(reference.reason):
                return ReferenceCommit(current, False, FaultCode.GIMBAL_FAULT)
            return ReferenceCommit(current, False, None)
        return ReferenceCommit(current, False, None)

    def _stow_evidence(self, state: PayloadState, now: float) -> bool:
        """Advance bounded stow evidence for the committed StowReference.

        Rate path: stow_reference_step each due step until inhibited at stow.
        Detailed plant: the armed switch reading True triggers inhibit; the
        tick completes once inhibit is confirmed with fresh feedback.
        """
        if not isinstance(state.reference, StowReference) or not self.stow_progress.armed:
            return False
        if self.stow_progress.complete:
            return True
        if self._rate_mode():
            assert isinstance(self.gimbal, GimbalRateActuator)
            with self.actuator_io_lock:
                step = self.gimbal.stow_reference_step(now)
            if isinstance(step, Err):
                self._record_actuator_failure(step.error, now)
                self._publish_fault(step.error, "bounded gimbal stow failed")
                self._latch_containment("bounded stow failed")
                return False
            if not step.value:
                return False
        else:
            with self.actuator_io_lock:
                switch = self.gimbal.read_stow_switch()
            if isinstance(switch, Err):
                self._publish_fault(switch.error, "stow switch read failed")
                return False
            if not switch.value:
                return False
            self._inhibit_motion("stow switch reached")
        health = self.actuator_safety.last_health
        last_fb = self.actuator_safety.last_feedback_s
        fresh = (
            last_fb is not None
            and math.isfinite(last_fb)
            and now - last_fb <= self.params.feedback_max_age_s
        )
        if fresh and health is not None and health.inhibit_confirmed:
            self.stow_progress.complete = True
        return self.stow_progress.complete

    def advance_outer(self, state: PayloadState, now: float) -> tuple[PayloadState, TickOutcome]:
        """Catch up the outer graph cadence to `now` in T_out steps.

        Each due tick dequeues at most one due CapturedVision (oldest first),
        reads ephemeris, steps the live graph when activated, commits the
        outcome, and drives the rate-mode actuator (or arms stow evidence).
        """
        dt = self.params.config.controller.outer.dt_s
        current = state
        command_issued = False
        if current.last_outer_s is None:
            origin = min(self.clock.monotonic_s(), now)
            current = replace(current, last_outer_s=origin)
            t = origin
        else:
            t = current.last_outer_s
        gap = now - t
        cap = self.params.config.controller.integrity.catchup_max_s
        if gap > cap:
            self._publish_fault(FaultCode.GIMBAL_FAULT, "outer catch-up cap exceeded")
            self._latch_containment("outer catch-up cap exceeded")
            t = now - cap
            current = replace(current, last_outer_s=t)
        while t + dt <= now + 1e-12:
            t = t + dt
            current = replace(current, last_outer_s=t)
            self._read_health()
            encoder = self._encoder_for_tick(t)
            if encoder is None:
                # A catch-up tick may not consume a current encoder sample and
                # relabel it with historical time. Keep the committed cursor at
                # the last processed tick; a later call may then replay this
                # historical interval when the physical sample arrives.
                if self.containment.local_latched:
                    current = replace(current, servo=self._invalidate_encoder(current.servo))
                current, _edge = self._drain_commands(current, t, None)
                continue
            vision: CapturedVision | None = None
            with self.state_lock:
                if self.vision_queue and self.vision_queue[0].sample.t_s <= t + 1e-9:
                    vision = self.vision_queue.popleft()
            if vision is not None and self._context_stale(vision.context, current):
                vision = None
            iss, eph_err = self._read_iss_at(t)
            if eph_err is not None:
                self._publish_fault(eph_err, "ephemeris read failed")
            current = replace(
                current,
                servo=replace(
                    current.servo,
                    encoder=replace(current.servo.encoder, last_theta_enc_rad=encoder.angle_rad),
                ),
            )
            if self.actuator_safety.recovery_pending_reset:
                current = replace(current, servo=self._fresh_servo_state(current.servo))
            stow_complete = self._stow_evidence(current, t)
            if current.graph is not None:
                inputs = self._tick_inputs(
                    current,
                    t,
                    timestamp_utc=self.clock.wall_clock_iso(),
                    encoder=encoder,
                    vision=vision,
                    navigation=iss,
                    stow_complete=stow_complete,
                )
                flagged = False
                if vision is not None and isinstance(current.graph, operate.State):
                    accepted = accept_vision(current.graph, inputs, self._params(current))
                    flagged = accepted is not None and accepted.sample.mode_flags != 0
                if flagged:
                    new_graph, outcome = runtime.step(current.graph, inputs, self._params(current))
                    current = replace(current, graph=new_graph)
                    current, issued = self._commit_outcome(current, outcome.outcome, t)
                    command_issued = command_issued or issued
                    current, _edge = self._drain_commands(
                        current, t, None, blocked_reason="flagged vision"
                    )
                else:
                    current, edge = self._drain_commands(current, t, inputs)
                    if not edge and current.graph is not None:
                        new_graph, outcome = runtime.step(
                            current.graph, inputs, self._params(current)
                        )
                        current = replace(current, graph=new_graph)
                        current, issued = self._commit_outcome(current, outcome.outcome, t)
                        command_issued = command_issued or issued
                self._publish_pointing(current, t, encoder, vision)
            else:
                current, _edge = self._drain_commands(current, t, None)
            if self._rate_mode():
                if isinstance(current.reference, StowReference):
                    pass
                elif self.containment.local_latched or isinstance(
                    current.reference, InhibitReference
                ):
                    reason = (
                        "containment" if self.containment.local_latched else "inhibit reference"
                    )
                    self._inhibit_motion(reason)
                    current = replace(current, servo=self._fresh_servo_state(current.servo))
                else:
                    rate = self.servo.reference_rate(
                        current.reference, encoder.angle_rad, detailed_plant=False
                    )
                    if isinstance(rate, Err):
                        self._publish_fault(rate.error, "invalid control reference")
                        self._latch_containment("invalid control reference")
                    else:
                        self._install(current)
                        self._write_rate(math.degrees(rate.value), t, current.control_revision)
                        current = replace(
                            current,
                            servo=replace(current.servo, commanded_rate_rad_s=rate.value),
                        )
        self._install(current)
        outcome_msg = TickOutcome(
            frame_id=0,
            fault=None,
            command_issued=command_issued,
            payload_graph=graph_name_of(current),
            payload_node=node_name_of(current),
        )
        return current, outcome_msg

    def advance_inner(self, state: PayloadState, now: float) -> PayloadState:
        """Catch up the inner servo cadence to `now` in T_in steps.

        Detailed plant: each tick reads the encoder, runs ServoController
        .inner_step against the committed reference, checks integrity, and
        writes leased torque. Contained or inhibit references never write
        torque -- the shell calls inhibit instead of commanding zero. Rate
        mode only records encoder feedback here; its actuation is the outer
        cadence rate write.
        """
        if self._rate_mode():
            # Production control is rate-commanded at outer cadence and does not
            # run the detailed-plant PI. Catch-up still records one encoder
            # sample so interleaved outer ticks have feedback at shutter.
            self._read_position()
            return state
        dt = self.servo.cfg.inner.dt_s
        current = state
        if current.servo.inner.last_inner_s is None:
            origin = min(self.clock.monotonic_s(), now)
            current = replace(
                current,
                servo=replace(
                    current.servo,
                    inner=replace(current.servo.inner, last_inner_s=origin),
                ),
            )
        t = current.servo.inner.last_inner_s
        assert t is not None
        gap = now - t
        cap = self.params.config.controller.integrity.catchup_max_s
        if gap > cap:
            self._publish_fault(FaultCode.GIMBAL_FAULT, "inner catch-up cap exceeded")
            self._latch_containment("inner catch-up cap exceeded")
            t = now - cap
            current = replace(
                current,
                servo=replace(
                    current.servo,
                    inner=replace(current.servo.inner, last_inner_s=t),
                ),
            )
        while t + dt <= now + 1e-12:
            t = t + dt
            pos = self._read_position()
            if not isinstance(pos, Ok):
                self._publish_fault(pos.error, "encoder unavailable")
                current = replace(
                    current,
                    servo=replace(current.servo, commanded_rate_rad_s=0.0),
                )
                break
            encoder = EncoderSample(
                sample_id=f"inner:{pos.value.sequence}:{pos.value.timestamp_s:.9f}",
                t_s=pos.value.timestamp_s,
                angle_rad=math.radians(pos.value.el_deg),
                angle_variance_rad2=self.params.config.controller.residual.encoder_variance_rad2,
            )
            if self.actuator_safety.recovery_pending_reset:
                current = replace(current, servo=self._fresh_servo_state(current.servo))
            enc_rate = 0.0
            prior_samples = current.servo.encoder.samples
            if current.servo.encoder.last_theta_enc_rad is not None and prior_samples:
                measured_dt_s = encoder.t_s - prior_samples[-1].t_s
                if measured_dt_s > 0.0:
                    enc_rate = (
                        encoder.angle_rad - current.servo.encoder.last_theta_enc_rad
                    ) / measured_dt_s
            tick: InnerTick = self.servo.inner_step(
                current.servo, t, encoder, current.reference, dt_s=dt
            )
            servo, integrity_fault = self._apply_integrity(current.servo, tick, enc_rate, t)
            current = replace(current, servo=servo)
            if integrity_fault is not None:
                self._publish_fault(integrity_fault, "pointing integrity trip")
                self._latch_containment("pointing integrity trip")
            if self.containment.local_latched or isinstance(current.reference, InhibitReference):
                self._inhibit_motion(
                    "contained" if self.containment.local_latched else "inhibit reference"
                )
                current = replace(current, servo=self._fresh_servo_state(current.servo))
            else:
                check = self.servo.reference_rate(
                    current.reference, encoder.angle_rad, detailed_plant=True
                )
                if isinstance(check, Err):
                    self._publish_fault(check.error, "invalid control reference")
                    self._latch_containment("invalid control reference")
                else:
                    self._install(current)
                    self._write_torque(tick.tau_nm, t, current.control_revision)
        self._install(current)
        return current

    def _apply_integrity(
        self,
        prior: ServoState,
        tick: InnerTick,
        enc_rate: float,
        now: float,
    ) -> tuple[ServoState, FaultCode | None]:
        """Run the integrity detector and stamp strikes onto tick state."""
        del now
        integrity = check_integrity(
            self.servo.cfg.integrity,
            tick.state.commanded_rate_rad_s,
            tick.state.encoder.measured_rate_rad_s,
            tick.tau_nm,
            enc_rate,
            prior.integrity.freeze_strikes,
        )
        updated = replace(
            tick.state,
            integrity=replace(tick.state.integrity, freeze_strikes=integrity.freeze_strikes),
        )
        return updated, integrity.fault

    @staticmethod
    def _fresh_servo_state(servo: ServoState) -> ServoState:
        """Discard dynamic PI/integrity memory; keep valid encoder history."""
        return replace(
            servo,
            inner=replace(servo.inner, integrator=0.0, last_tau_nm=0.0),
            integrity=replace(servo.integrity, freeze_strikes=0),
            commanded_rate_rad_s=0.0,
        )

    @staticmethod
    def _invalidate_encoder(servo: ServoState) -> ServoState:
        """Remove motion authority and the encoder baseline after a read fault."""
        return replace(
            servo,
            encoder=replace(
                servo.encoder,
                samples=(),
                last_theta_enc_rad=None,
                measured_rate_rad_s=0.0,
            ),
            commanded_rate_rad_s=0.0,
        )

    def _capture_context(self, latest: PayloadState) -> CaptureContext | None:
        """Stamp the capture context for the control-latest committed state.

        Inputs:
            latest: The committed state read under state_lock by the caller.

        Outputs:
            CaptureContext | None: The context tokens for `latest`, or None
                while unactivated. The caller reads containment.generation
                under the same lock that produced `latest`.
        """
        last = latest.activation.last
        if last is None:
            return None
        return CaptureContext(
            activation_key=last.key,
            policy_revision=latest.policy_revision,
            model_version="",
            containment_generation=self.containment.generation,
        )

    def _imaging_fault(
        self,
        code: FaultCode,
        detail: str,
        context: CaptureContext | None = None,
        state: PayloadState | None = None,
    ) -> bool:
        """Queue imaging containment for the control owner and publish the fault.

        The capture loop never calls the gimbal: a real imaging policy or HAL
        failure sets pending_fault (consumed once by _poll_local_faults, which
        latches containment on the next control poll) and publishes the
        original FaultEventMsg, both inside one state_lock window so the
        staleness check and the publication stay atomic. Global camera-control
        failures (setters, start, stop) pass no context: their physical
        effects persist across activations and always report. Frame-scoped
        acquire/drain results pass `context`: a stale completion is dropped
        with False and neither publishes nor queues anything, so an old
        activation's failure can never poison the newer one.

        Inputs:
            code: The original HAL or policy fault code.
            detail: Stage description for the fault record.
            context: Optional frame-scoped context to match atomically.
            state: Fallback committed state when the shell has none.

        Outputs:
            bool: True when the fault was queued and published as current;
                False when a provided context no longer matches.
        """
        with self.state_lock:
            if context is not None:
                shell = self.runtime_shell.state
                current = shell if shell is not None else state
                if current is None or self._context_stale(context, current):
                    return False
            self.capture_shell.pending_fault = code
            self._publish_fault(code, detail)
        return True

    def _publish_applied_policy(
        self,
        context: CaptureContext,
        policy: EffectivePolicy,
        state: PayloadState | None = None,
    ) -> bool:
        """Emit the compact applied-policy record once per current revision.

        The match, the revision mark, and the publish all run inside one
        state_lock window: a stale or superseded context records nothing, so
        a superseded application can never emit the old policy. An enabled
        policy additionally requires the containment latch to be clear; an
        explicitly requested off policy may be recorded while contained
        because off is factual. No lock is held across HAL calls.

        Inputs:
            context: Capture context stamped for this application.
            policy: The requested policy now in effect.
            state: Fallback committed state when the shell has none.

        Outputs:
            bool: True when the context is current (event emitted unless the
                revision was already recorded); False when stale.
        """
        with self.state_lock:
            shell = self.runtime_shell.state
            current = shell if shell is not None else state
            if current is None or self._capture_context(current) != context:
                return False
            if policy.imaging.acquisition_enabled and self.containment.local_latched:
                return False
            if self.capture_shell.applied_revision == context.policy_revision:
                return True
            self.capture_shell.applied_revision = context.policy_revision
            key = context.activation_key
            self._publish_telemetry(
                "imaging_policy",
                {
                    "policy_revision": context.policy_revision,
                    "activation_epoch": key.epoch,
                    "activation_sequence": key.sequence,
                    "acquisition_enabled": policy.imaging.acquisition_enabled,
                    "exposure_us": policy.imaging.exposure_us,
                    "gain_db": policy.imaging.gain_db,
                    "capture_interval_s": policy.imaging.capture_interval_s,
                    "duty_cycle": policy.imaging.duty_cycle,
                    "inference_enabled": policy.inference.enabled,
                    "every_n_frames": policy.inference.every_n_frames,
                },
            )
        return True

    def capture_once(self, state: PayloadState, now: float) -> tuple[PayloadState, TickOutcome]:
        """One planned capture cycle under the committed policy.

        The control-owned latest state and the capture context are stamped
        atomically before any settings/start/acquire work: a blocked call may
        span an activation, and both the off decision and the context tokens
        must honor that snapshot. Unactivated, contained, or disabled policy
        forces acquisition off -- no acquire/detect/drain runs. The complete
        policy is validated before any HAL call; settings apply only while
        stopped (a changed exposure/gain stops first, then sets exposure,
        gain, and restarts), with the context rechecked before and after
        every potentially blocking stage. plan_capture then decides WAIT,
        DRAIN, or CAPTURE; a captured frame runs process_frame. Returns the
        passed state unchanged -- capture never commits graph or servo state.
        """
        with self.state_lock:
            shell = self.runtime_shell.state
            latest = shell if shell is not None else state
            context = self._capture_context(latest)
            contained = self.containment.local_latched
        if not math.isfinite(now):
            self._imaging_fault(FaultCode.COMMAND_INVALID, "nonfinite capture time")
            return state, TickOutcome(0, FaultCode.COMMAND_INVALID, False)
        policy = latest.policy
        if context is None or contained or not policy.imaging.acquisition_enabled:
            if self.capture_shell.acquisition_on:
                stopped = self._stop_acquisition()
                if isinstance(stopped, Err):
                    return state, TickOutcome(0, stopped.error, False)
            # Only an explicitly requested off policy records as applied; a
            # forced stop of a still-enabled policy emits nothing.
            if context is not None and not policy.imaging.acquisition_enabled:
                self._publish_applied_policy(context, policy, state)
            return state, TickOutcome(0, None, False)
        limits = self.params.policy_limits
        valid = validate_policy(policy.imaging, policy.inference, limits)
        if isinstance(valid, Err):
            self._imaging_fault(valid.error, "imaging policy invalid")
            return state, TickOutcome(0, valid.error, False)
        capture = (policy.imaging.exposure_us, policy.imaging.gain_db)
        if self.capture_shell.acquisition_on and self.capture_shell.applied_capture != capture:
            stopped = self._stop_acquisition()
            if isinstance(stopped, Err):
                return state, TickOutcome(0, stopped.error, False)
        if not self.capture_shell.acquisition_on:
            if self._context_stale(context, self._latest(state)):
                return state, TickOutcome(0, None, False)
            exposure = self.sensor.set_exposure_us(capture[0])
            if isinstance(exposure, Err):
                self._imaging_fault(exposure.error, "imaging exposure apply failed")
                return state, TickOutcome(0, exposure.error, False)
            if self._context_stale(context, self._latest(state)):
                return state, TickOutcome(0, None, False)
            gain = self.sensor.set_gain_db(capture[1])
            if isinstance(gain, Err):
                self._imaging_fault(gain.error, "imaging gain apply failed")
                return state, TickOutcome(0, gain.error, False)
            if self._context_stale(context, self._latest(state)):
                return state, TickOutcome(0, None, False)
            started = self.sensor.start_acquisition()
            if isinstance(started, Err):
                self._imaging_fault(started.error, "imaging sensor start failed")
                return state, TickOutcome(0, started.error, False)
            self.capture_shell.acquisition_on = True
            self.capture_shell.applied_capture = capture
            if self._context_stale(context, self._latest(state)):
                stopped = self.sensor.stop_acquisition()
                if isinstance(stopped, Err):
                    self._imaging_fault(stopped.error, "imaging sensor stop failed")
                    return state, TickOutcome(0, stopped.error, False)
                self.capture_shell.acquisition_on = False
                self.capture_shell.applied_capture = None
                return state, TickOutcome(0, None, False)
        # One context-current application seam: emits the applied-policy
        # telemetry once per revision, for a fresh start and for an already
        # running camera, and confirms the context outlived the HAL work.
        if not self._publish_applied_policy(context, policy, state):
            return state, TickOutcome(0, None, False)
        planned = plan_capture(self.capture_shell.schedule, policy, context, now, limits)
        if isinstance(planned, Err):
            self._imaging_fault(planned.error, "imaging plan rejected")
            return state, TickOutcome(0, planned.error, False)
        self.capture_shell.schedule = planned.value.schedule
        if self._context_stale(context, self._latest(state)):
            return state, TickOutcome(0, None, False)
        decision = planned.value.decision
        if decision is CaptureDecision.WAIT:
            return state, TickOutcome(0, None, False)
        if decision is CaptureDecision.DRAIN:
            drained = self.sensor.drain_frame()
            if isinstance(drained, Err):
                faulted = self._imaging_fault(
                    drained.error, "imaging sensor buffer drain", context, state
                )
                return state, TickOutcome(0, drained.error if faulted else None, False)
            return state, TickOutcome(0, None, False)
        acquired = self.sensor.acquire_frame()
        if isinstance(acquired, Err):
            faulted = self._imaging_fault(acquired.error, "imaging sensor stall", context, state)
            return state, TickOutcome(0, acquired.error if faulted else None, False)
        return self.process_frame(acquired.value, state, now, context=context)

    def _stop_acquisition(self) -> Result[None, FaultCode]:
        """Stop the camera stream; only a confirmed stop clears applied state.

        A failed stop retains acquisition_on and the applied settings, queues
        the fault for control-owned containment, and returns the Err.
        """
        stopped = self.sensor.stop_acquisition()
        if isinstance(stopped, Err):
            self._imaging_fault(stopped.error, "imaging sensor stop failed")
            return stopped
        self.capture_shell.acquisition_on = False
        self.capture_shell.applied_capture = None
        return stopped

    def process_frame(
        self,
        raw: MosaicFrame,
        state: PayloadState,
        now: float,
        gimbal_pos: GimbalPosition | None = None,
        context: CaptureContext | None = None,
    ) -> tuple[PayloadState, TickOutcome]:
        """Preprocess, detect, and enqueue a context-stamped CapturedVision.

        Commits nothing to graph or servo state. After the (possibly blocking)
        detect the context tokens are rechecked against the passed state: a
        stale activation key or policy revision, or an engaged containment
        latch, drops the result before inference/product publication and
        enqueue. The accepted capture is counted once through record_capture;
        when inference is disabled or not due this frame returns fault-free
        without preprocessing, detection, products, or vision, so a skipped
        frame never becomes an empty observation or a miss. An explicit
        gimbal_pos injects the shutter-time actuator position for synchronous
        SIL/test callers that own no control loop.
        """
        if context is None:
            with self.state_lock:
                shell = self.runtime_shell.state
                latest = shell if shell is not None else state
                context = self._capture_context(latest)
            if context is None:
                return state, TickOutcome(raw.frame_id, None, False)
        else:
            latest = self._latest(state)
        if self._context_stale(context, latest):
            return state, TickOutcome(raw.frame_id, None, False)
        schedule, run_inference = record_capture(
            self.capture_shell.schedule, context, latest.policy.inference
        )
        self.capture_shell.schedule = schedule
        if not run_inference:
            return state, TickOutcome(raw.frame_id, None, False)
        stacked = stack_channels(np.asarray(raw.mosaic))
        if isinstance(stacked, Err):
            self._publish_fault(stacked.error, f"stack failed frame_id={raw.frame_id}")
            return state, self._fault_outcome(raw.frame_id, stacked.error, state)

        calibrated = calibrate_mosaic(stacked.value, self.calib)
        if isinstance(calibrated, Err):
            self._publish_fault(calibrated.error, f"calibration failed frame_id={raw.frame_id}")
            return state, self._fault_outcome(raw.frame_id, calibrated.error, state)

        normalized = normalize_dn(calibrated.value, self.sensor_cfg.bit_depth)
        selected = select_bands(
            normalized, self.sensor_cfg.channel_layout, self.inference_cfg.input_bands
        )
        if isinstance(selected, Err):
            self._publish_fault(selected.error, f"band select failed frame_id={raw.frame_id}")
            return state, self._fault_outcome(raw.frame_id, selected.error, state)

        enc_samples = self._encoder_snapshot()
        if gimbal_pos is not None:
            enc_samples = enc_samples + (
                EncoderSample(
                    sample_id=(
                        f"encoder:{gimbal_pos.sequence}"
                        if gimbal_pos.sequence > 0
                        else f"encoder:t:{gimbal_pos.timestamp_s:.9f}"
                    ),
                    t_s=gimbal_pos.timestamp_s,
                    angle_rad=math.radians(gimbal_pos.el_deg),
                    angle_variance_rad2=(
                        self.params.config.controller.residual.encoder_variance_rad2
                    ),
                ),
            )

        quality_flags = compute_quality_flags(
            selected.value,
            raw.exposure_us,
            raw.timestamp_utc,
            self.preprocessing_cfg,
        )

        shutter_t_s = raw.timestamp_s if raw.timestamp_s else now
        shutter_theta = self._encoder_angle_at(shutter_t_s, samples=enc_samples)
        if shutter_theta is not None and not math.isfinite(shutter_theta):
            shutter_theta = None
        iss, eph_err = self._read_iss_at(shutter_t_s)
        if eph_err is not None:
            self._publish_fault(eph_err, "ephemeris read failed")
        tile_gsd, gsd_nominal = self._tile_gsd_for_frame(iss, shutter_theta, enc_samples)
        if gsd_nominal:
            quality_flags = quality_flags | frozenset({FrameUsabilityTag.GSD_NOMINAL})

        processed = ProcessedFrameMsg(
            msg_type=MessageType.PROCESSED_FRAME,
            timestamp_utc=raw.timestamp_utc,
            frame_id=raw.frame_id,
            tensor=selected.value[np.newaxis, ...],
            quality_flags=quality_flags,
            tile_gsd_m=tile_gsd,
        )

        detect_result = self.detector.detect(processed)
        if isinstance(detect_result, Err):
            with self.state_lock:
                shell = self.runtime_shell.state
                current = shell if shell is not None else state
                stale = self._context_stale(context, current)
                if not stale:
                    self._publish_fault(
                        detect_result.error, f"detection failed frame_id={raw.frame_id}"
                    )
            if stale:
                return state, TickOutcome(
                    raw.frame_id, None, False, graph_name_of(current), node_name_of(current)
                )
            return state, self._fault_outcome(raw.frame_id, detect_result.error, state)
        inference = detect_result.value

        stored: tuple[str, str, int] | None = None
        pre_store = self._latest(state)
        if (
            not self._context_stale(context, pre_store)
            and pre_store.policy.inference.enabled
            and pre_store.policy.imaging.publish_products
        ):
            stored = self._store_mask_product(inference)

        sample = VisionSample(
            t_s=shutter_t_s,
            frame_id=str(raw.frame_id),
            z_v=None,
            p_cog=None,
            exposure_us=raw.exposure_us,
            blobs=tuple(inference.blobs),
            mode_flags=inference.mode_flags,
            iss=iss,
            theta_g_rad=shutter_theta,
        )
        with self.state_lock:
            shell = self.runtime_shell.state
            latest = shell if shell is not None else state
            if self._context_stale(context, latest):
                return state, TickOutcome(
                    raw.frame_id, None, False, graph_name_of(latest), node_name_of(latest)
                )
            context = replace(context, model_version=inference.model_version)
            self.bus.publish(inference)
            if stored is not None:
                self._publish_product_ref(*stored)
            self.vision_queue.append(CapturedVision(context=context, sample=sample))
        return state, TickOutcome(
            raw.frame_id, None, False, graph_name_of(latest), node_name_of(latest)
        )

    def _context_stale(self, context: CaptureContext, state: PayloadState) -> bool:
        """True when the capture context no longer matches the control-latest state."""
        last = state.activation.last
        if last is None or last.key != context.activation_key:
            return True
        if context.policy_revision != state.policy_revision:
            return True
        if context.containment_generation != self.containment.generation:
            return True
        return self.containment.local_latched

    def note_gimbal_feedback(self, position: GimbalPosition) -> None:
        """Record encoder feedback on a tick that does not capture."""
        self._record_encoder(position)

    def sample_feedback(self) -> Result[GimbalPosition, FaultCode]:
        """Take one control-owned feedback sample at the current clock.

        Public seam for the deterministic composition root (SIL): a non-grid
        shutter binding needs actual encoder evidence at the real clock time,
        so the control owner records one sample rather than letting the bind
        or capture paths touch the gimbal HAL.
        """
        return self._read_position()

    def _record_encoder(self, position: GimbalPosition) -> EncoderSample:
        """Record one valid feedback frame and return its estimator-domain sample."""
        sample_id = (
            f"encoder:{position.sequence}"
            if position.sequence > 0
            else f"encoder:t:{position.timestamp_s:.9f}"
        )
        sample = EncoderSample(
            sample_id=sample_id,
            t_s=position.timestamp_s,
            angle_rad=math.radians(position.el_deg),
            angle_variance_rad2=self.params.config.controller.residual.encoder_variance_rad2,
        )
        with self.state_lock:
            if all(
                existing.sample_id != sample.sample_id for existing in self.encoder_stream.samples
            ):
                self.encoder_stream.samples.append(sample)
                self._prune_consumed_ids()
        return sample

    def _prune_consumed_ids(self) -> None:
        """Drop consumption markers for samples no longer in the retained history."""
        retained = {sample.sample_id for sample in self.encoder_stream.samples}
        self.encoder_stream.consumed_ids &= retained

    def _encoder_for_tick(self, tick_s: float) -> EncoderSample | None:
        """Return the newest unconsumed sample whose device time belongs to this tick."""
        candidates = [
            sample
            for sample in self.encoder_stream.samples
            if sample.sample_id not in self.encoder_stream.consumed_ids
            and sample.t_s <= tick_s + 1.0e-12
        ]
        if not candidates:
            return None
        selected = max(candidates, key=lambda sample: (sample.t_s, sample.sample_id))
        self.encoder_stream.consumed_ids.add(selected.sample_id)
        self._prune_consumed_ids()
        return selected

    def _encoder_angle_at(
        self,
        t_s: float,
        *,
        max_span_s: float | None = None,
        samples: tuple[EncoderSample, ...] | None = None,
    ) -> float | None:
        """Interpolate a shutter angle only from a valid bounded sample bracket.

        ``samples`` defaults to the live control-owned deque (control callers);
        the capture loop always passes a locked tuple snapshot.
        """
        source = self.encoder_stream.samples if samples is None else samples
        ordered = sorted(source, key=lambda sample: sample.t_s)
        exact = [sample for sample in ordered if abs(sample.t_s - t_s) <= 1.0e-12]
        if exact:
            return exact[0].angle_rad
        before = [sample for sample in ordered if sample.t_s < t_s]
        after = [sample for sample in ordered if sample.t_s > t_s]
        if not before or not after:
            return None
        left = before[-1]
        right = after[0]
        span = right.t_s - left.t_s
        span_limit = (
            self.params.config.controller.residual.interpolation_span_max_s
            if max_span_s is None
            else max_span_s
        )
        if span <= 0.0 or span > span_limit:
            return None
        alpha = (t_s - left.t_s) / span
        return left.angle_rad + alpha * (right.angle_rad - left.angle_rad)

    def _encoder_rate_over_exposure_deg_per_s(
        self, raw: MosaicFrame, samples: tuple[EncoderSample, ...] | None = None
    ) -> float | None:
        """Mean elevation rate across the exposure from encoder brackets, or None."""
        dt_exp_s = raw.exposure_us * 1.0e-6
        if dt_exp_s <= 0.0:
            return None
        t_end = raw.timestamp_s
        t_start = t_end - dt_exp_s
        theta_end = self._encoder_angle_at(t_end, max_span_s=math.inf, samples=samples)
        theta_start = self._encoder_angle_at(t_start, max_span_s=math.inf, samples=samples)
        if theta_end is None or theta_start is None:
            return None
        return math.degrees((theta_end - theta_start) / dt_exp_s)

    def _smear_gimbal_rate_deg_per_s(
        self,
        raw: MosaicFrame,
        state: PayloadState,
        measured_rate_deg_per_s: float | None,
        samples: tuple[EncoderSample, ...] | None = None,
    ) -> tuple[float, SmearRateSource]:
        """Elevation rate selected by the payload app: measured, else encoder, else command.

        ``0.0`` is a valid measured, encoder, or commanded rate. Unknown is a
        missing measured value plus a failed encoder bracket, which falls back
        to the commanded rate and labels it COMMANDED. Quality flags do not
        raise MOTION_SMEAR from this rate.
        """
        if measured_rate_deg_per_s is not None:
            return measured_rate_deg_per_s, SmearRateSource.MEASURED
        encoder_rate = self._encoder_rate_over_exposure_deg_per_s(raw, samples)
        if encoder_rate is not None:
            return encoder_rate, SmearRateSource.ENCODER
        return math.degrees(state.servo.commanded_rate_rad_s), SmearRateSource.COMMANDED

    def _tile_gsd_for_frame(
        self,
        iss: IssSample | None,
        shutter_theta_rad: float | None,
        enc_samples: tuple[EncoderSample, ...] | None = None,
    ) -> tuple[np.ndarray, bool]:
        """Compute per-tile ground sampling distance, marking all reference fallbacks.

        The detector consumes row-major (tile_count, 2) lateral/along-track metres.
        Missing shutter angle or ephemeris uses the last finite encoder angle and a
        circular reference orbit derived from configured mean motion when possible.
        If those rays miss, the model reference GSD is supplied and tagged nominal.
        """
        cfg = self.inference_cfg
        ephemeris_cfg = self.params.config.ephemeris
        grid = (cfg.tile_rows, cfg.tile_cols)
        camera = CameraGeometry(
            width_px=self.sensor_cfg.width_px,
            height_px=self.sensor_cfg.height_px,
            pixel_pitch_m=self.sensor_cfg.pixel_um * 1.0e-6,
            focal_length_m=self.sensor_cfg.optics.focal_length_mm * 1.0e-3,
        )

        theta = shutter_theta_rad
        state_r: tuple[float, float, float] | None = None
        state_v: tuple[float, float, float] | None = None
        utc_s = self.clock.utc_s()
        epoch_utc_s = ephemeris_cfg.epoch_utc_s
        is_nominal = False
        if iss is not None and theta is not None:
            state_r, state_v, utc_s = iss.r_m, iss.v_m_s, iss.utc_s
        else:
            is_nominal = True
            if theta is None:
                source = self.encoder_stream.samples if enc_samples is None else enc_samples
                finite_samples = [sample for sample in source if math.isfinite(sample.angle_rad)]
                if finite_samples:
                    theta = max(finite_samples, key=lambda sample: sample.t_s).angle_rad
            if theta is not None and math.isfinite(theta):
                mean_motion_rad_s = ephemeris_cfg.mean_motion_rev_per_day * 2.0 * math.pi / 86_400.0
                if math.isfinite(mean_motion_rad_s) and mean_motion_rad_s > 0.0:
                    semi_major_m = (ephemeris_cfg.mu_m3_s2 / mean_motion_rad_s**2) ** (1.0 / 3.0)
                    altitude_m = semi_major_m - ephemeris_cfg.wgs84_a_m
                    nominal = nominal_iss_state(
                        altitude_m,
                        ephemeris_cfg.wgs84_a_m,
                        ephemeris_cfg.mu_m3_s2,
                    )
                    if nominal is not None:
                        state_r, state_v, epoch_utc_s = nominal
                        utc_s = epoch_utc_s

        grid_values: np.ndarray | None = None
        if (
            theta is not None
            and math.isfinite(theta)
            and state_r is not None
            and state_v is not None
        ):
            grid_values = tile_gsd_grid(
                theta,
                state_r,
                state_v,
                utc_s,
                epoch_utc_s,
                ephemeris_cfg.omega_earth_rad_s,
                ephemeris_cfg.wgs84_a_m,
                ephemeris_cfg.wgs84_f,
                camera,
                grid=grid,
            )
        if grid_values is None:
            is_nominal = True
            grid_values = np.full((*grid, 2), cfg.gsd_reference_m, dtype=np.float32)
        return grid_values.reshape(cfg.tile_rows * cfg.tile_cols, 2), is_nominal

    def _rate_mode(self) -> bool:
        """Return whether the injected actuator exposes the production rate path."""
        return isinstance(self.gimbal, GimbalRateActuator)

    def _read_position(self) -> Result[GimbalPosition, FaultCode]:
        """Read encoder feedback under the sole driver-I/O lock.

        A failed read immediately latches drive containment before the next torque
        write.  So does a nonfinite position or timestamp: an invalid sample
        must never enter the encoder history or the PI rate-fit ring, where it
        would poison a later otherwise healthy recovery.  The pure controller
        receives the hardware's sample timestamp separately, so no wall-clock
        arrival time is mistaken for an encoder time.
        """
        with self.actuator_io_lock:
            result = self.gimbal.read_position()
        if isinstance(result, Ok):
            position = result.value
            if not (math.isfinite(position.el_deg) and math.isfinite(position.timestamp_s)):
                self._latch_containment("invalid encoder sample")
                self._publish_fault(FaultCode.GIMBAL_ENCODER_INVALID, "invalid encoder sample")
                return Err(FaultCode.GIMBAL_ENCODER_INVALID)
            self._record_encoder(result.value)
            self.actuator_safety.last_feedback_s = result.value.timestamp_s
        else:
            self._record_actuator_failure(result.error, self.clock.monotonic_s())
            self._inhibit_motion("encoder unavailable")
            self._mark_containment_latched()
        return result

    def _inhibit_motion(self, reason: str) -> bool:
        """Request driver-side containment and fault if it cannot be confirmed."""
        with self.actuator_io_lock:
            result = self.gimbal.inhibit(reason)
        if isinstance(result, Err):
            self._publish_fault(result.error, f"gimbal inhibit unconfirmed: {reason}")
            self._mark_containment_latched()
            return False
        if not result.value.inhibit_confirmed:
            self._publish_fault(FaultCode.GIMBAL_FAULT, f"gimbal inhibit unconfirmed: {reason}")
            self._mark_containment_latched()
            return False
        self.actuator_safety.last_health = result.value
        self._publish_actuator_health(result.value, force=True)
        return True

    def _write_torque(self, tau_nm: float, now: float, control_revision: int) -> bool:
        """Write a leased torque command only while feedback and integrity are healthy."""
        if control_revision != self.runtime_shell.generation:
            return False
        if self.containment.local_latched:
            return self._inhibit_motion("motion inhibited by containment")
        health = self._read_health()
        if isinstance(health, Err) or not health.value.feedback_valid:
            self._latch_containment("invalid actuator feedback")
            self._publish_fault(FaultCode.GIMBAL_FAULT, "invalid actuator feedback")
            return False
        authority_now = self.clock.monotonic_s()
        last_feedback_s = health.value.last_feedback_s
        if (
            last_feedback_s is None
            or not math.isfinite(last_feedback_s)
            or authority_now - last_feedback_s > self.servo.cfg.integrity.feedback_max_age_s
        ):
            self._latch_containment("stale actuator feedback")
            self._publish_fault(FaultCode.GIMBAL_FAULT, "stale actuator feedback")
            return False
        self._publish_actuator_health(health.value)
        valid_until_s = authority_now + self.servo.cfg.integrity.command_authority_s
        with self.actuator_io_lock:
            send = self.gimbal.set_torque(tau_nm, valid_until_s)
        if isinstance(send, Err):
            self._record_actuator_failure(send.error, now)
            self._inhibit_motion("torque command failed")
            return False
        if self.actuator_safety.recovery_pending_reset:
            self._publish_actuator_recovery("recovered")
        self.actuator_safety.recovery_pending_reset = False
        self.actuator_safety.recovery_attempts = 0
        self.actuator_safety.recovery_window_start_s = None
        return True

    def _write_rate(self, rate_deg_per_s: float, now: float, control_revision: int) -> bool:
        """Write a leased production rate command after feedback/safety checks."""
        if not self._rate_mode():
            return False
        if control_revision != self.runtime_shell.generation:
            return False
        if self.containment.local_latched:
            return self._inhibit_motion("motion inhibited by containment")
        health = self._read_health()
        if isinstance(health, Err) or not health.value.feedback_valid:
            self._latch_containment("invalid actuator feedback")
            self._publish_fault(FaultCode.GIMBAL_FAULT, "invalid actuator feedback")
            return False
        authority_now = self.clock.monotonic_s()
        last_feedback_s = health.value.last_feedback_s
        if (
            last_feedback_s is None
            or not math.isfinite(last_feedback_s)
            or authority_now - last_feedback_s > self.servo.cfg.integrity.feedback_max_age_s
        ):
            self._latch_containment("stale actuator feedback")
            self._publish_fault(FaultCode.GIMBAL_STALE_FEEDBACK, "stale actuator feedback")
            return False
        self._publish_actuator_health(health.value)
        command = GimbalRateCommand(
            rate_deg_per_s=rate_deg_per_s,
            valid_until_s=authority_now + self.servo.cfg.integrity.command_authority_s,
        )
        with self.actuator_io_lock:
            if not isinstance(self.gimbal, GimbalRateActuator):
                return False
            send = self.gimbal.set_rate(command)
        if isinstance(send, Err):
            self._record_actuator_failure(send.error, now)
            self._inhibit_motion("rate command failed")
            return False
        if self.actuator_safety.recovery_pending_reset:
            self._publish_actuator_recovery("recovered")
        self.actuator_safety.recovery_pending_reset = False
        self.actuator_safety.recovery_attempts = 0
        self.actuator_safety.recovery_window_start_s = None
        return True

    def _record_actuator_failure(self, code: FaultCode, now: float) -> None:
        """Classify failures; only transient availability faults receive a retry budget."""
        transient = code is FaultCode.COMM_TIMEOUT
        safety = self.actuator_safety
        cfg = self.servo.cfg.integrity
        if not transient:
            self._mark_containment_latched()
            self._inhibit_motion("unclassified actuator failure")
            self._publish_fault(code, "unclassified actuator failure")
            return
        if (
            safety.recovery_window_start_s is None
            or now - safety.recovery_window_start_s > cfg.recovery_window_s
        ):
            safety.recovery_window_start_s = now
            safety.recovery_attempts = 0
        safety.recovery_attempts += 1
        safety.recovery_pending_reset = True
        self._publish_actuator_recovery("retrying")
        if safety.recovery_attempts > cfg.recovery_max_attempts:
            self._mark_containment_latched()
            self._publish_fault(code, "actuator transient recovery budget exhausted")

    def control_tick(self, state: PayloadState, now: float) -> PayloadState:
        """One control-owner cycle: activation drain, inner servo, outer graph.

        The single ordering seam shared by run() and the SIL harness: payload
        commands commit inside the outer tick (flagged-vision inhibit first),
        never on the inner cadence. Heartbeat and sync remain the caller's
        concern so SIL can publish them deterministically.
        """
        state = self.poll_activations(state, now)
        state = self.advance_inner(state, now)
        state, _outer = self.advance_outer(state, now)
        return state

    def run(self, stop_event: threading.Event) -> None:
        """Run the control worker + capture loop until stop_event is set.

        Boot always inhibits motion and publishes one SystemModeSyncRequestMsg;
        while unsynced the request repeats on the watchdog cadence. The control
        worker owns activation drain, commands, inner/outer advance, heartbeat,
        and every gimbal HAL call; the capture loop (this thread) may block on
        acquire/detect without ever committing graph or servo state.
        """
        self._inhibit_motion("boot")
        boot = self.initial_state()
        self._install(boot)
        heartbeat_seq = 0
        last_heartbeat = self.clock.monotonic_s()
        last_sync = self.clock.monotonic_s()
        self._publish_sync_request(boot)

        def control_loop() -> None:
            """The single control owner: drain, advance, heartbeat, sync."""
            nonlocal heartbeat_seq, last_heartbeat, last_sync
            dt = self.servo.cfg.inner.dt_s
            while not stop_event.is_set():
                now = self.clock.monotonic_s()
                snap = self.control_tick(self._latest(boot), now)
                if now - last_heartbeat >= self.fault_cfg.watchdog_interval_s:
                    self.bus.publish(
                        HeartbeatMsg(
                            msg_type=MessageType.HEARTBEAT,
                            timestamp_utc=self.clock.wall_clock_iso(),
                            subsystem="payload",
                            sequence=heartbeat_seq,
                        )
                    )
                    heartbeat_seq += 1
                    last_heartbeat = now
                if (
                    snap.activation.last is None
                    and now - last_sync >= self.fault_cfg.watchdog_interval_s
                ):
                    self._publish_sync_request(snap)
                    last_sync = now
                stop_event.wait(timeout=dt)

        control_thread = threading.Thread(target=control_loop, name="payload-control", daemon=True)
        control_thread.start()
        try:
            while not stop_event.is_set():
                now = self.clock.monotonic_s()
                snap = self._latest(boot)
                self.capture_once(snap, now)
                interval = snap.policy.imaging.capture_interval_s
                outer_dt = self.params.config.controller.outer.dt_s
                stop_event.wait(timeout=min(interval, outer_dt))
        finally:
            stop_event.set()
            control_thread.join(timeout=1.0)
            if isinstance(self.gimbal, GimbalRateActuator):
                # The production adapter owns the vendor transport and must
                # confirm inhibit before closing it.  Detailed-plant SIL has
                # no shutdown surface and keeps its existing teardown path.
                self._inhibit_motion("payload app stopping")
                shutdown = self.gimbal.shutdown()
                if isinstance(shutdown, Err):
                    self._publish_fault(shutdown.error, "gimbal shutdown failed")
            self._stop_acquisition()

    def _publish_sync_request(self, state: PayloadState) -> None:
        """Publish the boot/watchdog SystemModeSyncRequestMsg."""
        last = state.activation.last
        self.bus.publish(
            SystemModeSyncRequestMsg(
                msg_type=MessageType.SYSTEM_MODE_SYNC_REQUEST,
                timestamp_utc=self.clock.wall_clock_iso(),
                subscriber="payload",
                expected_epoch=self.activation_epoch,
                request_id=str(uuid.uuid4()),
                last_sequence=last.key.sequence if last is not None else None,
            )
        )

    def _store_mask_product(self, inference: InferenceResultMsg) -> tuple[str, str, int] | None:
        """Persist a compact uint8 mask thumbnail; return ref fields on success.

        Only stores: the caller rechecks the capture context before publishing
        the ProductRefMsg, so a blocked store cannot publish a stale product.
        """
        mask = np.asarray(inference.mask, dtype=np.float32)
        if mask.ndim != 2 or mask.size == 0:
            return None
        step = max(1, mask.shape[0] // 32, mask.shape[1] // 32)
        thumb = (np.clip(mask[::step, ::step], 0.0, 1.0) * 255.0).astype(np.uint8)
        data = thumb.tobytes()
        item_id = f"mask_thumb_{inference.frame_id}"
        result = self.storage.store(item_id, data, DownlinkPriority.SCIENCE_PRODUCT)
        if not isinstance(result, Ok):
            return None
        return result.value, item_id, len(data)

    def _publish_product_ref(self, entry_id: str, item_id: str, byte_len: int) -> None:
        """Publish the ProductRefMsg for a stored mask product."""
        self.bus.publish(
            ProductRefMsg(
                msg_type=MessageType.PRODUCT_REF,
                timestamp_utc=self.clock.wall_clock_iso(),
                entry_id=entry_id,
                priority=DownlinkPriority.SCIENCE_PRODUCT,
                item_id=item_id,
                byte_len=byte_len,
            )
        )

    def _publish_gimbal_command(
        self,
        mode: GimbalCommandMode,
        el_value_deg: float,
        state: PayloadState,
        reason: str,
    ) -> None:
        """Publish the pose-actuation audit record with activation identity."""
        last = state.activation.last
        self.bus.publish(
            GimbalCommandMsg(
                msg_type=MessageType.GIMBAL_COMMAND,
                timestamp_utc=self.clock.wall_clock_iso(),
                frame_id=0,
                mode=mode,
                el_value_deg=el_value_deg,
                payload_graph=graph_name_of(state),
                payload_node=node_name_of(state),
                activation_epoch=last.key.epoch if last is not None else "",
                activation_sequence=last.key.sequence if last is not None else -1,
                reason=reason,
            )
        )

    def _publish_telemetry(
        self, event_name: str, payload: dict[str, str | int | float | bool | None]
    ) -> None:
        """Publish a compact payload TelemetryEventMsg."""
        self.bus.publish(
            TelemetryEventMsg(
                msg_type=MessageType.TELEMETRY_EVENT,
                timestamp_utc=self.clock.wall_clock_iso(),
                subsystem="payload",
                event_name=event_name,
                payload=payload,
            )
        )

    def _publish_actuator_recovery(self, state: str) -> None:
        """Expose bounded transient-recovery state without treating it as integrity."""
        self.bus.publish(
            TelemetryEventMsg(
                msg_type=MessageType.TELEMETRY_EVENT,
                timestamp_utc=self.clock.wall_clock_iso(),
                subsystem="payload",
                event_name="gimbal_actuator_recovery",
                payload={
                    "state": state,
                    "attempt": self.actuator_safety.recovery_attempts,
                },
            )
        )

    def _publish_actuator_health(self, health: GimbalHealth, force: bool = False) -> None:
        """Publish compact, persistent actuator-health evidence for FDIR and ground."""
        now = self.clock.monotonic_s()
        last = self.actuator_safety.last_health_publish_s
        if not force and last is not None and now - last < self.fault_cfg.watchdog_interval_s:
            return
        self.actuator_safety.last_health_publish_s = now
        self.actuator_safety.last_health = health
        self.bus.publish(
            TelemetryEventMsg(
                msg_type=MessageType.TELEMETRY_EVENT,
                timestamp_utc=self.clock.wall_clock_iso(),
                subsystem="payload",
                event_name="gimbal_actuator_health",
                payload={
                    "feedback_valid": health.feedback_valid,
                    "last_feedback_s": (
                        math.nan if health.last_feedback_s is None else health.last_feedback_s
                    ),
                    "command_valid_until_s": (
                        math.nan
                        if health.command_valid_until_s is None
                        else health.command_valid_until_s
                    ),
                    "inhibited": health.inhibited,
                    "inhibit_confirmed": health.inhibit_confirmed,
                    "controller_status_bits": health.controller_status_bits,
                    "motor_on": health.motor_on,
                    "closed_loop": health.closed_loop,
                    "duty_credit_s": (
                        -1.0 if health.duty_credit_s is None else health.duty_credit_s
                    ),
                    "duty_locked_out": health.duty_locked_out,
                    "watchdog_gate_confirmed": health.watchdog_gate_confirmed,
                    "time_mapping_valid": health.time_mapping_valid,
                    "requested_rate_deg_per_s": (
                        math.nan
                        if health.requested_rate_deg_per_s is None
                        else health.requested_rate_deg_per_s
                    ),
                    "quantized_rate_deg_per_s": (
                        math.nan
                        if health.quantized_rate_deg_per_s is None
                        else health.quantized_rate_deg_per_s
                    ),
                },
            )
        )

    def _read_iss_at(self, monotonic_t: float) -> tuple[IssSample | None, FaultCode | None]:
        """Read ISS ECI at the UTC corresponding to monotonic_t. Err is published by caller."""
        utc = self.clock.utc_s() + (monotonic_t - self.clock.monotonic_s())
        result = self.ephemeris.read_state(utc)
        if isinstance(result, Err):
            return None, result.error
        value = result.value
        return IssSample(r_m=value.r_m, v_m_s=value.v_m_s, utc_s=value.epoch_utc_s), None

    def _ack_command(
        self, command: RoutedCommandMsg, status: AckStatus, fault: FaultCode, detail: str
    ) -> None:
        """Publish an execution CommandAckMsg for a routed payload command."""
        self.bus.publish(
            CommandAckMsg(
                msg_type=MessageType.COMMAND_ACK,
                timestamp_utc=self.clock.wall_clock_iso(),
                status=status,
                command_id=command.command_id,
                source=command.source,
                seq=command.seq,
                fault_code=fault,
                detail=detail,
            )
        )

    def _publish_fault(self, code: FaultCode, detail: str) -> None:
        """Publish a FaultEventMsg from the payload subsystem onto the bus."""
        self.bus.publish(
            FaultEventMsg(
                msg_type=MessageType.FAULT_EVENT,
                timestamp_utc=self.clock.wall_clock_iso(),
                fault_code=code,
                subsystem="payload",
                detail=detail,
            )
        )

    def _fault_outcome(self, frame_id: int, code: FaultCode, state: PayloadState) -> TickOutcome:
        """Build a TickOutcome for a frame that faulted before control ran."""
        return TickOutcome(
            frame_id=frame_id,
            fault=code,
            command_issued=False,
            payload_graph=graph_name_of(state),
            payload_node=node_name_of(state),
        )


_CONTAINING_FAULTS: frozenset[FaultCode] = frozenset(
    {
        FaultCode.GIMBAL_FAULT,
        FaultCode.GIMBAL_RUNAWAY,
        FaultCode.GIMBAL_SAFETY_TIMEOUT,
        FaultCode.GIMBAL_STALE_FEEDBACK,
        FaultCode.GIMBAL_WATCHDOG_UNCONFIRMED,
        FaultCode.INFERENCE_NAN,
    }
)
