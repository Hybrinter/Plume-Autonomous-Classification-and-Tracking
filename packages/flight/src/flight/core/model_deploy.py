"""Core model-deployment service: stage validation + ACTIVATE with automatic rollback.

The core half of the model upload chain (spec Section 6; iss_iface owns reassembly). It consumes:

  - ModelStagedMsg (a reassembled artifact was stored): it fetches the bytes via the injected
    StorageReader, verifies the SHA-256 against the announced digest, and parses the JSON
    manifest. On success the deploy state becomes STAGED; a digest/parse failure raises
    MODEL_CORRUPT and leaves the active model untouched.
  - a routed ACTIVATE_MODEL command: it runs load-validation + a first-frame sanity check on the
    staged artifact -- modeled here as verifying the manifest's I/O contract matches the flight
    inference contract (input (1, C, H, W), output (1, 1, H, W)), since onnxruntime is not present
    in this repo. On success the staged model becomes ACTIVE (the previous becomes the rollback);
    on failure the service AUTOMATICALLY ROLLS BACK -- the previously active model stays active,
    the state becomes ROLLBACK_AVAILABLE, and a MODEL_CORRUPT fault is raised. ModelDeployStateMsg
    is telemetered on every transition.

Contains:
  - DeployState: mutable active/rollback/staged bookkeeping + ModelDeployState.
  - parse_manifest / contract_ok: pure manifest parsing + I/O-contract validation.
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
class StagedModel:
    """A validated, staged artifact awaiting activation."""

    entry_id: str
    version: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


@dataclass(slots=True)
class DeployState:
    """Mutable model-deployment bookkeeping.

    Fields:
        state: The lifecycle state (ACTIVE / STAGED / ROLLBACK_AVAILABLE).
        active_version: The currently active model identifier.
        rollback_version: The previous model retained for rollback (None at first boot).
        staged: The staged model awaiting ACTIVATE, or None.
    """

    state: ModelDeployState = ModelDeployState.ACTIVE
    active_version: str = "factory"
    rollback_version: str | None = None
    staged: StagedModel | None = None


@dataclass(slots=True, frozen=True)
class ParsedManifest:
    """A parsed, type-coerced model-upload manifest."""

    version: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


def parse_manifest(blob: bytes) -> ParsedManifest | None:
    """Parse a model-upload manifest (JSON bytes), or None if malformed.

    Args:
        blob: The reassembled artifact bytes (a JSON manifest in this SIL-modeled form).

    Returns:
        A ParsedManifest, or None if the bytes are not valid JSON / not an object / missing or
        non-numeric version/input_shape/output_shape fields.
    """
    try:
        data = json.loads(blob.decode("utf-8"))
    except ValueError, UnicodeDecodeError:
        return None
    if not isinstance(data, dict) or "version" not in data:
        return None
    raw_in = data.get("input_shape")
    raw_out = data.get("output_shape")
    if not isinstance(raw_in, list) or not isinstance(raw_out, list):
        return None
    try:
        input_shape = tuple(int(v) for v in raw_in)
        output_shape = tuple(int(v) for v in raw_out)
    except TypeError, ValueError:
        return None
    return ParsedManifest(str(data["version"]), input_shape, output_shape)


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

    def _expected_contract(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return the (input, output) shape contract derived from the inference config."""
        h, w = self.inference_cfg.input_height_px, self.inference_cfg.input_width_px
        return (1, len(self.inference_cfg.input_bands), h, w), (1, 1, h, w)

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
            input_shape=manifest.input_shape,
            output_shape=manifest.output_shape,
        )
        self.state.state = ModelDeployState.STAGED
        self._publish_state(self.state.staged.version, "model staged and validated")

    def _handle_activate(self, command: RoutedCommandMsg) -> None:
        """Activate the staged model with a contract sanity check; auto-rollback on failure."""
        staged = self.state.staged
        if staged is None:
            self._ack(command, False, "no staged model to activate")
            return
        expected_in, expected_out = self._expected_contract()
        if contract_ok(staged.input_shape, staged.output_shape, expected_in, expected_out):
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
