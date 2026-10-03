"""SIL integration: chunked model upload -> reassemble -> stage -> activate -> rollback."""

import base64
import json
import zlib

from flight.libs.commands import build_tc_packet
from flight.libs.config import PactConfig
from flight.libs.messages import ModelDeployStateMsg
from flight.libs.time import ManualClock
from flight.libs.types import ModelDeployState
from sim.scene import build_frames, plume_detector
from sim.sil import SilHarness, build_sil_system

_KEY = b"sil-test-key-0000000000000000000"
_INF = PactConfig().inference
_H = _INF.input_height_px
_W = _INF.input_width_px
_ROWS, _COLS = _INF.tile_rows, _INF.tile_cols
_TH, _TW = _H // _ROWS, _W // _COLS


def _manifest(version: str, classifier_channels: int, segmentor_channels: int = 3) -> bytes:
    """Conditioned pair manifest. Three channels match flight; other counts fail activate."""

    def entry(kind: str, channels: int) -> dict[str, object]:
        bands = list(_INF.input_bands)
        while len(bands) < channels:
            bands.append(f"EXTRA_{len(bands)}")
        bands = bands[:channels]
        return {
            "arch": "pactnet" if kind == "classifier" else "dilatenet",
            "sha256": "a" * 64,
            "input_names": ["image", "gsd"],
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

    return json.dumps(
        {
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
    ).encode("utf-8")


def _chunk_packets(blob: bytes, base_seq: int) -> list[bytes]:
    """Split blob into two signed UPLOAD_MODEL_CHUNK TC packets with sequential seqs."""
    crc = zlib.crc32(blob) & 0xFFFFFFFF
    mid = max(1, len(blob) // 2)
    parts = [blob[:mid], blob[mid:]]
    packets = []
    for i, part in enumerate(parts):
        packets.append(
            build_tc_packet(
                "UPLOAD_MODEL_CHUNK",
                {
                    "chunk_index": i,
                    "total_chunks": len(parts),
                    "data_b64": base64.b64encode(part).decode("ascii"),
                    "crc32": crc,
                },
                "ground",
                base_seq + i,
                _KEY,
                apid=1,
            )
        )
    return packets


def test_model_upload_activate_then_rollback() -> None:
    """A good model uploads + activates; a contract-bad model uploads + auto-rolls-back."""
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(40),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    harness = SilHarness(system)
    deploy_states = system.bus.subscribe(ModelDeployStateMsg)

    now = 0.0

    def advance(steps: int) -> None:
        nonlocal now
        for _ in range(steps):
            now += 1.0
            harness.step(now)

    def activate(seq: int) -> None:
        system.station.enqueue(
            build_tc_packet("ACTIVATE_MODEL", {"version": "v"}, "ground", seq, _KEY, apid=1)
        )

    def deploy_state() -> ModelDeployState:
        """Read the current deploy state (fresh local avoids mypy identity-narrowing)."""
        return system.apps.model_deploy.state.state

    # --- good model: upload -> stage -> activate -> ACTIVE ---
    for pkt in _chunk_packets(_manifest("v2", classifier_channels=3), base_seq=1):
        system.station.enqueue(pkt)
    advance(4)  # ingest + route + reassemble + stage
    assert deploy_state() is ModelDeployState.STAGED
    activate(seq=3)
    advance(4)
    assert deploy_state() is ModelDeployState.ACTIVE
    assert system.apps.model_deploy.state.active_version == "v2"

    # --- bad model: upload -> stage -> activate -> auto-rollback ---
    for pkt in _chunk_packets(
        _manifest("v3", classifier_channels=4, segmentor_channels=4), base_seq=4
    ):
        system.station.enqueue(pkt)
    advance(4)
    assert deploy_state() is ModelDeployState.STAGED
    activate(seq=6)
    advance(4)
    assert deploy_state() is ModelDeployState.ROLLBACK_AVAILABLE
    assert system.apps.model_deploy.state.active_version == "v2"  # stayed on the last good model

    seen = [m.state for m in _drain(deploy_states)]
    assert ModelDeployState.STAGED in seen
    assert ModelDeployState.ACTIVE in seen
    assert ModelDeployState.ROLLBACK_AVAILABLE in seen


def _drain(subscription: object) -> list:  # type: ignore[type-arg]
    """Drain all pending messages from a subscription into a list."""
    out: list = []  # type: ignore[type-arg]
    while not subscription.empty():  # type: ignore[attr-defined]
        out.append(subscription.get_nowait())  # type: ignore[attr-defined]
    return out
