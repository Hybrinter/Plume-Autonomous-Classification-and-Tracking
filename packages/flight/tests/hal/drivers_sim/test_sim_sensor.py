"""Tests for the simulated imaging sensor."""

import numpy as np
from flight.hal.drivers_sim import SimSensor
from flight.libs.types import Err, FaultCode, MosaicFrame, Ok


def _frame(frame_id: int) -> MosaicFrame:
    """Build a minimal MosaicFrame with a (8, 8) uint16 mosaic plane."""
    mosaic = np.zeros((8, 8), dtype=np.uint16)  # np.ndarray[uint16, (H, W)]
    return MosaicFrame(
        timestamp_utc="2026-05-31T00:00:00.000Z",
        timestamp_s=float(frame_id),
        frame_id=frame_id,
        mosaic=mosaic,
        exposure_us=10_000.0,
        gain_db=0.0,
    )


def test_returns_frames_in_order() -> None:
    """acquire_frame yields each frame once, in order."""
    sensor = SimSensor([_frame(1), _frame(2)])
    first = sensor.acquire_frame()
    second = sensor.acquire_frame()
    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert first.value.frame_id == 1
    assert second.value.frame_id == 2


def test_stalls_when_exhausted() -> None:
    """acquire_frame returns Err(CAMERA_STALL) once frames are exhausted."""
    sensor = SimSensor([_frame(1)])
    sensor.acquire_frame()
    result = sensor.acquire_frame()
    assert isinstance(result, Err)
    assert result.error is FaultCode.CAMERA_STALL


def test_control_calls_succeed() -> None:
    """Tuning and acquisition control calls are no-ops that succeed."""
    sensor = SimSensor([])
    assert isinstance(sensor.set_exposure_us(5000.0), Ok)
    assert isinstance(sensor.set_gain_db(1.0), Ok)
    assert isinstance(sensor.start_acquisition(), Ok)
    assert isinstance(sensor.stop_acquisition(), Ok)


def test_load_next_bootstrap_from_empty() -> None:
    """An empty constructor stalls until load_next supplies a slot."""
    sensor = SimSensor([])
    stalled = sensor.acquire_frame()
    assert isinstance(stalled, Err)
    assert stalled.error is FaultCode.CAMERA_STALL
    sensor.load_next(_frame(9))
    result = sensor.acquire_frame()
    assert isinstance(result, Ok)
    assert result.value.frame_id == 9


def test_load_next_overwrites_unread_slot() -> None:
    """A second load_next discards the unread slot."""
    sensor = SimSensor([])
    sensor.load_next(_frame(1))
    sensor.load_next(_frame(2))
    result = sensor.acquire_frame()
    assert isinstance(result, Ok)
    assert result.value.frame_id == 2
    stalled = sensor.acquire_frame()
    assert isinstance(stalled, Err)
    assert stalled.error is FaultCode.CAMERA_STALL


def test_unread_scripted_count_tracks_constructor_list() -> None:
    """unread_scripted_count ignores the live slot and drops as frames are acquired."""
    sensor = SimSensor([_frame(1), _frame(2)])
    sensor.load_next(_frame(99))
    assert sensor.unread_scripted_count() == 2
    sensor.acquire_frame()
    assert sensor.unread_scripted_count() == 2
    sensor.acquire_frame()
    assert sensor.unread_scripted_count() == 1
    sensor.acquire_frame()
    assert sensor.unread_scripted_count() == 0


def test_load_next_prefers_slot_over_scripted_list() -> None:
    """The live slot is returned before remaining constructor frames."""
    sensor = SimSensor([_frame(1), _frame(2)])
    sensor.load_next(_frame(99))
    first = sensor.acquire_frame()
    second = sensor.acquire_frame()
    assert isinstance(first, Ok)
    assert isinstance(second, Ok)
    assert first.value.frame_id == 99
    assert second.value.frame_id == 1
