"""Tests for PayloadApp encoder-stream retention and consumption bookkeeping."""

import math

import numpy as np
from flight.hal.drivers_sim import SimGimbal, SimIssEphemeris, SimSensor
from flight.hal.interfaces import GimbalPosition
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.time import ManualClock
from flight.libs.types import DownlinkPriority, FaultCode, Ok, Result
from flight.payload.app import EncoderStream, PayloadApp
from flight.payload.calibration_io import build_identity_calibration
from flight.payload.inference import DetectorBackend, ScriptedDetector

_ENCODER_HISTORY_MAX = 4096


class _MemStorage:
    """In-memory StorageWriter double for encoder-stream tests."""

    def store(
        self, item_id: str, data: bytes, priority: DownlinkPriority
    ) -> Result[str, FaultCode]:
        """Return a synthetic entry id without persisting data."""
        return Ok(f"entry_{item_id}")


def _build_app(detector: DetectorBackend) -> PayloadApp:
    """Assemble a PayloadApp over sim drivers and a fresh bus."""
    cfg = PactConfig()
    bus = MessageBus()
    clock = ManualClock()
    gimbal = SimGimbal(clock=clock, cfg=cfg.gimbal, inner_dt_s=cfg.controller.inner.dt_s)
    sensor = SimSensor([])
    eph = SimIssEphemeris(clock=clock, cfg=cfg.ephemeris)
    calib = build_identity_calibration(cfg.sensor.height_px, cfg.sensor.width_px)
    app = PayloadApp.from_config(
        cfg, sensor, gimbal, eph, detector, bus, clock, calib, _MemStorage()
    )
    app.lock_gate.engaged = False
    return app


def _gimbal_pos(sequence: int, t_s: float, el_deg: float = 0.0) -> GimbalPosition:
    """Build a timestamped GimbalPosition with a nonzero sequence ID."""
    return GimbalPosition(el_deg=el_deg, timestamp_s=t_s, sequence=sequence)


def _assert_consumed_ids_bounded(stream: EncoderStream) -> None:
    """Assert consumption metadata stays within the retained sample history."""
    assert len(stream.consumed_ids) <= len(stream.samples)
    assert len(stream.samples) <= _ENCODER_HISTORY_MAX
    retained = {sample.sample_id for sample in stream.samples}
    assert stream.consumed_ids <= retained


def _plume_detector() -> ScriptedDetector:
    """Minimal scripted detector for PayloadApp construction."""
    mask = np.zeros((1544, 2064), dtype=np.float32)
    return ScriptedDetector(mask, confidence_gate=0.55, min_blob_area_px=15)


def test_consumed_ids_stay_bounded_past_history_capacity() -> None:
    """Recording and consuming past maxlen keeps consumed_ids within retained samples."""
    app = _build_app(_plume_detector())
    dt_s = 0.02
    n_samples = 5000
    for i in range(n_samples):
        seq = i + 1
        t_s = seq * dt_s
        app._record_encoder(_gimbal_pos(seq, t_s, el_deg=float(seq)))
        _assert_consumed_ids_bounded(app.encoder_stream)
        if seq % 3 == 0:
            selected = app._encoder_for_tick(t_s)
            assert selected is not None
            assert selected.sample_id == f"encoder:{seq}"
            _assert_consumed_ids_bounded(app.encoder_stream)
    assert len(app.encoder_stream.samples) == _ENCODER_HISTORY_MAX
    assert len(app.encoder_stream.consumed_ids) <= _ENCODER_HISTORY_MAX


def test_retained_sample_is_not_reconsumed() -> None:
    """A sample still in history cannot be selected twice."""
    app = _build_app(_plume_detector())
    app._record_encoder(_gimbal_pos(1, 0.02, el_deg=1.0))
    app._record_encoder(_gimbal_pos(2, 0.04, el_deg=2.0))
    first = app._encoder_for_tick(0.02)
    assert first is not None
    assert first.sample_id == "encoder:1"
    retry = app._encoder_for_tick(0.02)
    assert retry is None
    later = app._encoder_for_tick(0.04)
    assert later is not None
    assert later.sample_id == "encoder:2"


def test_sequence_reset_reuses_id_after_eviction() -> None:
    """A reused sequence number is selectable once the prior row left the deque."""
    app = _build_app(_plume_detector())
    n_samples = _ENCODER_HISTORY_MAX + 100
    for seq in range(1, n_samples + 1):
        t_s = seq * 0.02
        app._record_encoder(_gimbal_pos(seq, t_s))
        app._encoder_for_tick(t_s)
    assert "encoder:1" not in {sample.sample_id for sample in app.encoder_stream.samples}
    reset = _gimbal_pos(1, (n_samples + 1) * 0.02, el_deg=99.0)
    app._record_encoder(reset)
    selected = app._encoder_for_tick(reset.timestamp_s)
    assert selected is not None
    assert selected.sample_id == "encoder:1"
    assert math.isclose(selected.angle_rad, math.radians(99.0))


def test_duplicate_record_does_not_grow_samples() -> None:
    """Recording the same sample_id twice leaves one row in the deque."""
    app = _build_app(_plume_detector())
    pos = _gimbal_pos(7, 0.14, el_deg=3.0)
    app._record_encoder(pos)
    app._record_encoder(pos)
    assert len(app.encoder_stream.samples) == 1
    assert app.encoder_stream.samples[0].sample_id == "encoder:7"


def test_out_of_order_timestamps_pick_newest_eligible() -> None:
    """Selection uses the newest unconsumed sample with device time at or before the tick."""
    app = _build_app(_plume_detector())
    app._record_encoder(_gimbal_pos(1, 0.10, el_deg=1.0))
    app._record_encoder(_gimbal_pos(2, 0.30, el_deg=3.0))
    app._record_encoder(_gimbal_pos(3, 0.20, el_deg=2.0))
    selected = app._encoder_for_tick(0.25)
    assert selected is not None
    assert selected.sample_id == "encoder:3"
    assert math.isclose(selected.t_s, 0.20)
