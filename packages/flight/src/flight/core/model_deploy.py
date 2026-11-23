"""Core model-deployment service: pair staging, activation, and rollback.

It verifies uploaded pair bytes and manifest metadata, then activates only pairs
that match the configured dynamic-batch image/GSD contract, tile geometry, and
preprocessing metadata. Failed activation preserves the previous active pair.

Satisfies: REQ-AIML-HIGH-004, REQ-COMM-MODEL-001.
"""

from __future__ import annotations

# stdlib
import hashlib
import json
import math
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
from flight.payload.inference.contract import verify_conditioned_shapes

SUBSYSTEM = "model_deploy"
_ACTIVATE_MODEL = "ACTIVATE_MODEL"


@dataclass(slots=True, frozen=True)
class ArtifactContract:
    """I/O tensor shapes for one network in an inference pair."""

    input_shape: tuple[int | None, ...]
    gsd_input_shape: tuple[int | None, ...]
    output_shape: tuple[int | None, ...]
    input_names: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class StagedModel:
    """A validated, staged classifier+segmentor pair awaiting activation."""

    entry_id: str
    version: str
    grid: tuple[int, int]
    frame_hw: tuple[int, int]
    tile_hw: tuple[int, int]
    gsd_reference_m: float
    norm: str
    conditioning: str
    gsd_encoding: str
    band_names: tuple[str, ...]
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
    """A parsed classifier+segmentor pair manifest and flight metadata."""

    version: str
    grid: tuple[int, int]
    frame_hw: tuple[int, int]
    tile_hw: tuple[int, int]
    gsd_reference_m: float
    norm: str
    conditioning: str
    gsd_encoding: str
    band_names: tuple[str, ...]
    classifier: ArtifactContract
    segmentor: ArtifactContract


def _shape(raw: object) -> tuple[int | None, ...] | None:
    """Accept arrays of positive integer dimensions and JSON null wildcards."""
    if not isinstance(raw, list) or not raw:
        return None
    if any(value is not None and (type(value) is not int or value <= 0) for value in raw):
        return None
    return tuple(raw)


def _parse_contract(
    raw: object,
    kind: str,
    channels: int,
    tile_hw: tuple[int, int],
    reference: float,
    norm: str,
    conditioning: str,
    encoding: str,
    bands: tuple[str, ...],
) -> ArtifactContract | None:
    """Parse and validate one conditioned network entry."""
    if not isinstance(raw, dict):
        return None
    image = _shape(raw.get("input_shape"))
    gsd = _shape(raw.get("gsd_input_shape"))
    output = _shape(raw.get("output_shape"))
    names = raw.get("input_names")
    types = raw.get("input_types")
    digest = raw.get("sha256")
    raw_reference = raw.get("gsd_reference_m")
    if image is None or gsd is None or output is None:
        return None
    if (
        not isinstance(names, list)
        or len(names) != 2
        or any(not isinstance(name, str) for name in names)
        or set(names) != {"image", "gsd"}
        or len(set(names)) != 2
        or not isinstance(types, dict)
        or types != {"image": "float32", "gsd": "float32"}
        or raw.get("output_type") != "float32"
        or type(raw_reference) not in (int, float)
        or raw_reference != reference
        or raw.get("norm") != norm
        or raw.get("conditioning") != conditioning
        or raw.get("gsd_encoding") != encoding
        or raw.get("band_names") != list(bands)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in digest)
    ):
        return None
    checked = verify_conditioned_shapes(image, gsd, output, channels, tile_hw, kind)
    if isinstance(checked, Err):
        return None
    return ArtifactContract(image, gsd, output, ("image", "gsd"))


def _positive_int_pair(raw: object) -> tuple[int, int] | None:
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or any(type(value) is not int or value <= 0 for value in raw)
    ):
        return None
    return raw[0], raw[1]


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
    if not isinstance(data, dict) or not isinstance(data.get("version"), str):
        return None
    grid = _positive_int_pair(data.get("grid"))
    frame_hw = _positive_int_pair(data.get("frame_hw"))
    tile_hw = _positive_int_pair(data.get("tile_hw"))
    reference = data.get("gsd_reference_m")
    norm, conditioning, encoding = (
        data.get("norm"),
        data.get("conditioning"),
        data.get("gsd_encoding"),
    )
    raw_classifier = data.get("classifier")
    raw_bands = raw_classifier.get("band_names") if isinstance(raw_classifier, dict) else None
    bands = (
        tuple(raw_bands)
        if isinstance(raw_bands, list)
        and raw_bands
        and all(isinstance(band, str) and band for band in raw_bands)
        and len(set(raw_bands)) == len(raw_bands)
        else None
    )
    if (
        grid is None
        or frame_hw is None
        or tile_hw is None
        or not isinstance(reference, (int, float))
        or isinstance(reference, bool)
        or not math.isfinite(reference)
        or reference <= 0
        or bands is None
        or not isinstance(norm, str)
        or not isinstance(conditioning, str)
        or not isinstance(encoding, str)
    ):
        return None
    channels = len(bands)
    classifier = _parse_contract(
        data.get("classifier"),
        "classifier",
        channels,
        tile_hw,
        float(reference),
        norm,
        conditioning,
        encoding,
        bands,
    )
    segmentor = _parse_contract(
        data.get("segmentor"),
        "segmentor",
        channels,
        tile_hw,
        float(reference),
        norm,
        conditioning,
        encoding,
        bands,
    )
    if classifier is None or segmentor is None:
        return None
    return ParsedManifest(
        data["version"],
        grid,
        frame_hw,
        tile_hw,
        float(reference),
        norm,
        conditioning,
        encoding,
        bands,
        classifier,
        segmentor,
    )


def contract_ok(
    input_shape: tuple[int | None, ...],
    output_shape: tuple[int | None, ...],
    expected_input: tuple[int | None, ...],
    expected_output: tuple[int | None, ...],
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
        rows, cols = self.inference_cfg.tile_rows, self.inference_cfg.tile_cols
        tile_hw = (h // rows, w // cols)
        shared_input = (None, len(self.inference_cfg.input_bands), *tile_hw)
        return (
            ArtifactContract(shared_input, (None, 2), (None, 1), ("image", "gsd")),
            ArtifactContract(shared_input, (None, 2), (None, 1, *tile_hw), ("image", "gsd")),
        )

    def _metadata_matches_config(self, manifest: ParsedManifest) -> bool:
        """Require pair geometry and preprocessing metadata to match this flight config."""
        cfg = self.inference_cfg
        rows, cols = cfg.tile_rows, cfg.tile_cols
        frame_hw = (cfg.input_height_px, cfg.input_width_px)
        if frame_hw[0] % rows or frame_hw[1] % cols:
            return False
        return (
            manifest.grid == (rows, cols)
            and manifest.frame_hw == frame_hw
            and manifest.tile_hw == (frame_hw[0] // rows, frame_hw[1] // cols)
            and manifest.gsd_reference_m == cfg.gsd_reference_m
            and manifest.band_names == tuple(cfg.input_bands)
            and manifest.norm == "unit"
            and manifest.conditioning == "film-log-gsd-v1"
            and manifest.gsd_encoding == "ln_metres_over_reference_lateral_along"
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
            grid=manifest.grid,
            frame_hw=manifest.frame_hw,
            tile_hw=manifest.tile_hw,
            gsd_reference_m=manifest.gsd_reference_m,
            norm=manifest.norm,
            conditioning=manifest.conditioning,
            gsd_encoding=manifest.gsd_encoding,
            band_names=manifest.band_names,
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
        pair_ok = (
            self._metadata_matches_config(
                ParsedManifest(
                    staged.version,
                    staged.grid,
                    staged.frame_hw,
                    staged.tile_hw,
                    staged.gsd_reference_m,
                    staged.norm,
                    staged.conditioning,
                    staged.gsd_encoding,
                    staged.band_names,
                    staged.classifier,
                    staged.segmentor,
                )
            )
            and staged.classifier == expected_classifier
            and staged.segmentor == expected_segmentor
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
