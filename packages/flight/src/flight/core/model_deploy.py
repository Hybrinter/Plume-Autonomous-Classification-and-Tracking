"""Core model-deployment service: stage validation + ACTIVATE with automatic rollback.

The core half of the model upload chain (spec Section 6; iss_iface owns reassembly). It consumes:

  - ModelStagedMsg (a reassembled classifier+segmentor pair bundle was stored): it fetches the
    bytes via the injected StorageReader, verifies the SHA-256 against the announced digest, and
    parses the JSON manifest. The manifest must name both a classifier contract and a segmentor
    contract. On success the deploy state becomes STAGED; a digest/parse failure raises
    MODEL_CORRUPT and leaves the active pair untouched.
  - a routed ACTIVATE_MODEL command: it runs load-validation + a first-frame sanity check on the
    staged pair -- modeled here as verifying both I/O contracts match flight (shared input
    (1, C, H, W); classifier output (1, 1); segmentor output (1, 1, H, W)), since onnxruntime is
    not present in this repo. Both contracts must match. On success the staged pair becomes
    ACTIVE (the previous pair becomes the rollback); on failure the service AUTOMATICALLY ROLLS
    BACK -- the previously active pair stays active, the state becomes ROLLBACK_AVAILABLE, and a
    MODEL_CORRUPT fault is raised. ModelDeployStateMsg is telemetered on every transition.

Contains:
  - ArtifactContract: one network's input/output shapes.
  - DeployState: mutable active/rollback/staged bookkeeping + ModelDeployState.
  - parse_manifest / contract_ok: pure pair-manifest parsing + I/O-contract validation.
  - ModelDeployService: from_config(); tick(); run().

Satisfies: REQ-AIML-HIGH-004, REQ-COMM-MODEL-001.
"""

from __future__ import annotations

# stdlib
import hashlib
import json
import threading
from dataclasses import dataclass

# internal
from flight.hal.interfaces import StorageReader
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import FaultConfig, InferenceConfig, PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    FaultEventMsg,
    HeartbeatMsg,
    ModelDeployStateMsg,
    ModelStagedMsg,
    RoutedCommandMsg,
)
from flight.libs.time import Clock
from flight.libs.types import AckStatus, Err, FaultCode, MessageType, ModelDeployState

SUBSYSTEM = "model_deploy"
_ACTIVATE_MODEL = "ACTIVATE_MODEL"


@dataclass(slots=True, frozen=True)
class ArtifactContract:
    """I/O tensor shapes for one network in an inference pair."""

    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class StagedModel:
    """A validated, staged classifier+segmentor pair awaiting activation."""

    entry_id: str
    version: str
    classifier: ArtifactContract
    segmentor: ArtifactContract


@dataclass(slots=True)
class DeployState:
    """Mutable model-deployment bookkeeping.

    Fields:
        state: The lifecycle state (ACTIVE / STAGED / ROLLBACK_AVAILABLE).
        active_version: The currently active model identifier.
        rollback_version: The previous model retained for rollback (None at first boot).
        staged: The staged inference pair awaiting ACTIVATE, or None.
    """

    state: ModelDeployState = ModelDeployState.ACTIVE
    active_version: str = "factory"
    rollback_version: str | None = None
    staged: StagedModel | None = None


@dataclass(slots=True, frozen=True)
class ParsedManifest:
    """A parsed, type-coerced classifier+segmentor upload manifest."""

    version: str
    classifier: ArtifactContract
    segmentor: ArtifactContract


def _parse_contract(raw: object) -> ArtifactContract | None:
    """Parse one network's input_shape/output_shape object, or None if malformed."""
    if not isinstance(raw, dict):
        return None
    raw_in = raw.get("input_shape")
    raw_out = raw.get("output_shape")
    if not isinstance(raw_in, list) or not isinstance(raw_out, list):
        return None
    try:
        input_shape = tuple(int(v) for v in raw_in)
        output_shape = tuple(int(v) for v in raw_out)
    except TypeError, ValueError:
        return None
    return ArtifactContract(input_shape, output_shape)


def parse_manifest(blob: bytes) -> ParsedManifest | None:
    """Parse a pair-upload manifest (JSON bytes), or None if malformed.

    Args:
        blob: The reassembled pair-bundle bytes (a JSON manifest in this SIL-modeled form).

    Returns:
        A ParsedManifest, or None if the bytes are not valid JSON, are not an object, lack
        version, or lack a well-formed classifier or segmentor contract. A single-network
        legacy manifest is malformed.
    """
    try:
        data = json.loads(blob.decode("utf-8"))
    except ValueError, UnicodeDecodeError:
        return None
    if not isinstance(data, dict) or "version" not in data:
        return None
    classifier = _parse_contract(data.get("classifier"))
    segmentor = _parse_contract(data.get("segmentor"))
    if classifier is None or segmentor is None:
        return None
    return ParsedManifest(str(data["version"]), classifier, segmentor)


def contract_ok(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    expected_input: tuple[int, ...],
    expected_output: tuple[int, ...],
) -> bool:
    """Return True iff the manifest I/O shapes match the flight inference contract (pure)."""
    return input_shape == expected_input and output_shape == expected_output


@dataclass(frozen=True)
class ModelDeployService:
    """Core service: validate staged uploads and activate them with automatic rollback."""

    inference_cfg: InferenceConfig
    fault_cfg: FaultConfig
    bus: MessageBus
    clock: Clock
    storage_reader: StorageReader
    staged_sub: Subscription[ModelStagedMsg]
    commands: Subscription[RoutedCommandMsg]
    state: DeployState

    @staticmethod
    def from_config(
        cfg: PactConfig, bus: MessageBus, clock: Clock, storage_reader: StorageReader
    ) -> ModelDeployService:
        """Assemble a ModelDeployService subscribing to staged-model + routed-command messages.

        Args:
            cfg: Top-level PactConfig (inference for the I/O contract; fault for the heartbeat).
            bus: The shared MessageBus to subscribe to / publish onto.
            clock: Injected Clock (real or manual).
            storage_reader: The StorageReader used to fetch a staged artifact's bytes.

        Returns:
            A ModelDeployService in the ACTIVE state with the factory model.
        """
        return ModelDeployService(
            inference_cfg=cfg.inference,
            fault_cfg=cfg.fault,
            bus=bus,
            clock=clock,
            storage_reader=storage_reader,
            staged_sub=bus.subscribe(ModelStagedMsg),
            commands=bus.subscribe(RoutedCommandMsg),
            state=DeployState(),
        )

    def _expected_pair(self) -> tuple[ArtifactContract, ArtifactContract]:
        """Return (classifier, segmentor) contracts derived from the inference config."""
        h, w = self.inference_cfg.input_height_px, self.inference_cfg.input_width_px
        shared_input = (1, len(self.inference_cfg.input_bands), h, w)
        return (
            ArtifactContract(shared_input, (1, 1)),
            ArtifactContract(shared_input, (1, 1, h, w)),
        )

    def tick(self) -> None:
        """Process staged-model announcements then routed ACTIVATE_MODEL commands."""
        while not self.staged_sub.empty():
            self._handle_staged(self.staged_sub.get_nowait())
        while not self.commands.empty():
            command = self.commands.get_nowait()
            if command.target == SUBSYSTEM and command.command_id == _ACTIVATE_MODEL:
                self._handle_activate(command)

    def _handle_staged(self, msg: ModelStagedMsg) -> None:
        """Validate a staged artifact (digest + manifest) and move to STAGED, or fault."""
        read = self.storage_reader.read(msg.entry_id)
        if isinstance(read, Err):
            self._fault(FaultCode.MODEL_CORRUPT, f"staged artifact unreadable: {msg.entry_id}")
            return
        blob = read.value
        if hashlib.sha256(blob).hexdigest() != msg.sha256:
            self._fault(FaultCode.MODEL_CORRUPT, "staged artifact digest mismatch")
            return
        manifest = parse_manifest(blob)
        if manifest is None:
            self._fault(FaultCode.MODEL_CORRUPT, "staged artifact manifest malformed")
            return
        self.state.staged = StagedModel(
            entry_id=msg.entry_id,
            version=manifest.version,
            classifier=manifest.classifier,
            segmentor=manifest.segmentor,
        )
        self.state.state = ModelDeployState.STAGED
        self._publish_state(self.state.staged.version, "model staged and validated")

    def _handle_activate(self, command: RoutedCommandMsg) -> None:
        """Activate the staged model with a contract sanity check; auto-rollback on failure."""
        staged = self.state.staged
        if staged is None:
            self._ack(command, False, "no staged model to activate")
            return
        expected_classifier, expected_segmentor = self._expected_pair()
        pair_ok = contract_ok(
            staged.classifier.input_shape,
            staged.classifier.output_shape,
            expected_classifier.input_shape,
            expected_classifier.output_shape,
        ) and contract_ok(
            staged.segmentor.input_shape,
            staged.segmentor.output_shape,
            expected_segmentor.input_shape,
            expected_segmentor.output_shape,
        )
        if pair_ok:
            self.state.rollback_version = self.state.active_version
            self.state.active_version = staged.version
            self.state.staged = None
            self.state.state = ModelDeployState.ACTIVE
            self._publish_state(self.state.active_version, "model activated")
            self._ack(command, True, f"activated {staged.version}")
        else:
            # First-frame sanity / load validation failed: keep the previous model active.
            self.state.staged = None
            self.state.state = ModelDeployState.ROLLBACK_AVAILABLE
            self._fault(
                FaultCode.MODEL_CORRUPT,
                f"activation of {staged.version} failed sanity check; rolled back",
            )
            self._publish_state(self.state.active_version, f"rolled back from {staged.version}")
            self._ack(
                command, False, f"activation failed; rolled back to {self.state.active_version}"
            )

    def _publish_state(self, version: str, detail: str) -> None:
        """Publish the current ModelDeployState as telemetry."""
        self.bus.publish(
            ModelDeployStateMsg(
                msg_type=MessageType.MODEL_DEPLOY,
                timestamp_utc=self.clock.wall_clock_iso(),
                state=self.state.state,
                version=version,
                detail=detail,
            )
        )

    def _fault(self, code: FaultCode, detail: str) -> None:
        """Publish a FaultEventMsg from the model-deploy subsystem."""
        self.bus.publish(
            FaultEventMsg(
                msg_type=MessageType.FAULT_EVENT,
                timestamp_utc=self.clock.wall_clock_iso(),
                fault_code=code,
                subsystem=SUBSYSTEM,
                detail=detail,
            )
        )

    def _ack(self, command: RoutedCommandMsg, accepted: bool, detail: str) -> None:
        """Publish an execution CommandAckMsg for a routed ACTIVATE_MODEL command."""
        self.bus.publish(
            CommandAckMsg(
                msg_type=MessageType.COMMAND_ACK,
                timestamp_utc=self.clock.wall_clock_iso(),
                status=AckStatus.ACCEPTED if accepted else AckStatus.REJECTED,
                command_id=command.command_id,
                source=command.source,
                seq=command.seq,
                fault_code=FaultCode.NONE if accepted else FaultCode.MODEL_CORRUPT,
                detail=detail,
            )
        )

    def run(self, stop_event: threading.Event) -> None:
        """Run the deploy loop until stop_event is set, emitting periodic heartbeats.

        Args:
            stop_event: threading.Event; the loop exits cleanly once it is set.
        """
        sequence = 0
        last_heartbeat = self.clock.monotonic_s()
        while not stop_event.is_set():
            self.tick()
            now = self.clock.monotonic_s()
            if now - last_heartbeat >= self.fault_cfg.watchdog_interval_s:
                self.bus.publish(
                    HeartbeatMsg(
                        msg_type=MessageType.HEARTBEAT,
                        timestamp_utc=self.clock.wall_clock_iso(),
                        subsystem=SUBSYSTEM,
                        sequence=sequence,
                    )
                )
                sequence += 1
                last_heartbeat = now
            stop_event.wait(timeout=self.fault_cfg.watchdog_interval_s)
