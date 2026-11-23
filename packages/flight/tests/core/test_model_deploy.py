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

_INF = PactConfig().inference
_H = _INF.input_height_px
_W = _INF.input_width_px
_C = len(_INF.input_bands)
_ROWS = _INF.tile_rows
_COLS = _INF.tile_cols
_TH, _TW = _H // _ROWS, _W // _COLS


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


def _manifest(
    version: str,
    *,
    classifier_channels: int = _C,
    segmentor_channels: int = _C,
    omit: str | None = None,
) -> bytes:
    """Build a pair-manifest blob. Default channels match the flight contract."""

    def entry(kind: str, channels: int) -> dict[str, object]:
        bands = list(_INF.input_bands)
        while len(bands) < channels:
            bands.append(f"EXTRA_{len(bands)}")
        bands = bands[:channels]
        return {
            "arch": "pactnet" if kind == "classifier" else "dilatenet",
            "sha256": "a" * 64,
            "input_names": ["gsd", "image"],
            "input_types": {"image": "float32", "gsd": "float32"},
            "output_type": "float32",
            "input_shape": [None, channels, _TH, _TW],
            "gsd_input_shape": [None, 2],
            "output_shape": [None, 1] if kind == "classifier" else [None, 1, _TH, _TW],
            "gsd_reference_m": _INF.gsd_reference_m,
            "norm": "unit",
            "conditioning": "film-log-gsd-v1",
            "gsd_encoding": "ln_metres_over_reference_lateral_along",
            "band_names": bands,
            "gsd_min_m": [10.0, 10.0],
            "gsd_max_m": [50.0, 50.0],
        }

    data: dict[str, object] = {
        "version": version,
        "grid": [_ROWS, _COLS],
        "frame_hw": [_H, _W],
        "tile_hw": [_TH, _TW],
        "gsd_reference_m": _INF.gsd_reference_m,
        "norm": "unit",
        "conditioning": "film-log-gsd-v1",
        "gsd_encoding": "ln_metres_over_reference_lateral_along",
        "classifier": entry("classifier", classifier_channels),
        "segmentor": entry("segmentor", segmentor_channels),
    }
    if omit is not None:
        del data[omit]
    return json.dumps(data).encode("utf-8")


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
    """parse_manifest extracts both contracts; contract_ok compares shapes."""
    parsed = parse_manifest(_manifest("v2"))
    assert parsed is not None
    assert parsed.classifier.input_shape == (None, _C, _TH, _TW)
    assert parsed.classifier.gsd_input_shape == (None, 2)
    assert parsed.classifier.output_shape == (None, 1)
    assert parsed.segmentor.output_shape == (None, 1, _TH, _TW)
    assert contract_ok(
        parsed.segmentor.input_shape,
        parsed.segmentor.output_shape,
        (None, _C, _TH, _TW),
        (None, 1, _TH, _TW),
    )
    assert parse_manifest(b"not json") is None


def test_parse_manifest_rejects_single_network() -> None:
    """A classifier-only, segmentor-only, or legacy single-graph blob is malformed."""
    assert parse_manifest(_manifest("v2", omit="classifier")) is None
    assert parse_manifest(_manifest("v2", omit="segmentor")) is None
    legacy = json.dumps(
        {
            "version": "v2",
            "input_shape": [1, _C, _H, _W],
            "output_shape": [1, 1, _H, _W],
        }
    ).encode("utf-8")
    assert parse_manifest(legacy) is None


def test_parse_manifest_rejects_malformed_dimension_types() -> None:
    """JSON booleans, strings, and floats are not coerced into graph dimensions."""
    for malformed in (1, "1", 1.0, True):
        data = json.loads(_manifest("v2"))
        data["classifier"]["input_shape"][0] = malformed
        assert parse_manifest(json.dumps(data).encode()) is None


def test_parse_manifest_accepts_input_name_order_independently() -> None:
    """A valid pair may declare the two named inputs in either order."""
    parsed = parse_manifest(_manifest("v2"))
    assert parsed is not None
    assert parsed.classifier.input_names == ("image", "gsd")


def test_parse_manifest_rejects_non_object_network_entries() -> None:
    """Null or non-object classifier and segmentor entries fail cleanly."""
    malformed: object
    for field in ("classifier", "segmentor"):
        for malformed in (None, [], "network"):
            data = json.loads(_manifest("v2"))
            data[field] = malformed
            assert parse_manifest(json.dumps(data).encode()) is None


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
    blob = _manifest("v2")
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
    blob = _manifest("v3", classifier_channels=4, segmentor_channels=4)
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


def test_activate_bad_segmentor_contract_rolls_back() -> None:
    """A pair whose segmentor contract fails rolls back the whole pair."""
    storage = _MemStorageReader()
    blob = _manifest("v3", classifier_channels=4, segmentor_channels=4)
    storage.put("e1", blob)
    svc, bus = _service(storage)
    _stage(bus, "e1", blob)
    svc.tick()
    _activate(bus)
    svc.tick()
    assert svc.state.state is ModelDeployState.ROLLBACK_AVAILABLE
    assert svc.state.active_version == "factory"


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
