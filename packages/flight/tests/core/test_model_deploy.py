"""ModelDeployService tests: staging validation, activation, and automatic rollback."""

import hashlib
import json

from flight.core.model_deploy import ModelDeployService, contract_ok, parse_manifest
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    FaultEventMsg,
    ModelDeployStateMsg,
    ModelStagedMsg,
    RoutedCommandMsg,
)
from flight.libs.time import ManualClock
from flight.libs.types import AckStatus, Err, FaultCode, MessageType, ModelDeployState, Ok, Result


class _MemStorageReader:
    """In-memory StorageReader double."""

    def __init__(self) -> None:
        """Start empty."""
        self.items: dict[str, bytes] = {}

    def put(self, entry_id: str, data: bytes) -> None:
        """Seed an entry."""
        self.items[entry_id] = data

    def read(self, entry_id: str) -> Result[bytes, FaultCode]:
        """Read an entry's bytes or report STORAGE_CORRUPT when missing."""
        if entry_id in self.items:
            return Ok(self.items[entry_id])
        return Err(FaultCode.STORAGE_CORRUPT)


def _manifest(version: str, channels: int = 4) -> bytes:
    """Build a manifest blob; channels=4 matches the flight contract, others fail it."""
    return json.dumps(
        {
            "version": version,
            "input_shape": [1, channels, 256, 256],
            "output_shape": [1, 1, 256, 256],
        }
    ).encode("utf-8")


def _service(storage: _MemStorageReader) -> tuple[ModelDeployService, MessageBus]:
    """Build a ModelDeployService over an in-memory storage reader and a fresh bus."""
    bus = MessageBus()
    svc = ModelDeployService.from_config(PactConfig(), bus, ManualClock(), storage)
    return svc, bus


def _stage(bus: MessageBus, entry_id: str, blob: bytes) -> None:
    """Publish a ModelStagedMsg for a stored blob."""
    bus.publish(
        ModelStagedMsg(
            msg_type=MessageType.MODEL_STAGED,
            timestamp_utc="t",
            entry_id=entry_id,
            sha256=hashlib.sha256(blob).hexdigest(),
            version="",
        )
    )


def _activate(bus: MessageBus, seq: int = 1) -> None:
    """Publish a routed ACTIVATE_MODEL command targeting the deploy service."""
    bus.publish(
        RoutedCommandMsg(
            msg_type=MessageType.ROUTED_COMMAND,
            timestamp_utc="t",
            target="model_deploy",
            command_id="ACTIVATE_MODEL",
            params={"version": "v2"},
            source="ground",
            seq=seq,
        )
    )


def test_parse_manifest_and_contract() -> None:
    """parse_manifest extracts typed shapes; contract_ok compares against the expected contract."""
    parsed = parse_manifest(_manifest("v2"))
    assert parsed is not None
    assert parsed.input_shape == (1, 4, 256, 256)
    assert contract_ok(parsed.input_shape, parsed.output_shape, (1, 4, 256, 256), (1, 1, 256, 256))
    assert parse_manifest(b"not json") is None


def test_staging_valid_manifest_goes_staged() -> None:
    """A digest-verified, well-formed staged artifact transitions to STAGED."""
    storage = _MemStorageReader()
    blob = _manifest("v2")
    storage.put("e1", blob)
    svc, bus = _service(storage)
    states = bus.subscribe(ModelDeployStateMsg)
    _stage(bus, "e1", blob)
    svc.tick()
    assert svc.state.state is ModelDeployState.STAGED
    assert states.get_nowait().state is ModelDeployState.STAGED


def test_staging_digest_mismatch_faults() -> None:
    """A staged artifact whose bytes do not match the announced digest raises MODEL_CORRUPT."""
    storage = _MemStorageReader()
    storage.put("e1", _manifest("v2"))
    svc, bus = _service(storage)
    faults = bus.subscribe(FaultEventMsg)
    # Announce a digest for different bytes.
    bus.publish(
        ModelStagedMsg(
            msg_type=MessageType.MODEL_STAGED,
            timestamp_utc="t",
            entry_id="e1",
            sha256=hashlib.sha256(b"other").hexdigest(),
            version="",
        )
    )
    svc.tick()
    assert faults.get_nowait().fault_code is FaultCode.MODEL_CORRUPT
    assert svc.state.state is ModelDeployState.ACTIVE  # unchanged


def test_activate_good_model_goes_active() -> None:
    """Activating a contract-valid staged model makes it ACTIVE and acks ACCEPTED."""
    storage = _MemStorageReader()
    blob = _manifest("v2", channels=4)
    storage.put("e1", blob)
    svc, bus = _service(storage)
    acks = bus.subscribe(CommandAckMsg)
    _stage(bus, "e1", blob)
    svc.tick()
    _activate(bus)
    svc.tick()
    assert svc.state.state is ModelDeployState.ACTIVE
    assert svc.state.active_version == "v2"
    assert any(a.status is AckStatus.ACCEPTED for a in _drain(acks))


def test_activate_bad_contract_rolls_back() -> None:
    """Activating a staged model that fails the I/O contract auto-rolls-back and faults."""
    storage = _MemStorageReader()
    blob = _manifest("v3", channels=3)  # wrong channel count -> contract fails
    storage.put("e1", blob)
    svc, bus = _service(storage)
    acks = bus.subscribe(CommandAckMsg)
    faults = bus.subscribe(FaultEventMsg)
    _stage(bus, "e1", blob)
    svc.tick()
    _activate(bus)
    svc.tick()
    assert svc.state.state is ModelDeployState.ROLLBACK_AVAILABLE
    assert svc.state.active_version == "factory"  # stayed on the previous model
    assert any(a.status is AckStatus.REJECTED for a in _drain(acks))
    assert any(f.fault_code is FaultCode.MODEL_CORRUPT for f in _drain(faults))


def test_activate_without_staged_rejected() -> None:
    """ACTIVATE_MODEL with nothing staged is rejected."""
    svc, bus = _service(_MemStorageReader())
    acks = bus.subscribe(CommandAckMsg)
    _activate(bus)
    svc.tick()
    assert any(a.status is AckStatus.REJECTED for a in _drain(acks))


def _drain(sub: object) -> list:  # type: ignore[type-arg]
    """Drain a subscription into a list."""
    out: list = []  # type: ignore[type-arg]
    while not sub.empty():  # type: ignore[attr-defined]
        out.append(sub.get_nowait())  # type: ignore[attr-defined]
    return out
