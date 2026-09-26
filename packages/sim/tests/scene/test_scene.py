"""Tests for SIL plume scene generation."""

import numpy as np
from flight.libs.messages import ProcessedFrameMsg
from flight.libs.types import BAND_ORDER, MessageType, Ok
from sim.scene.plume import (
    DETECTOR_HEIGHT_PX,
    DETECTOR_WIDTH_PX,
    PLANE_HEIGHT_PX,
    PLANE_WIDTH_PX,
    build_frames,
    plume_detector,
)


def test_build_frames_count_and_shape() -> None:
    """build_frames returns N frames each a (3, H, W) uint16 RGB stack."""
    frames = build_frames(3)
    assert len(frames) == 3
    planes = np.asarray(frames[0].planes)
    assert planes.shape == (len(BAND_ORDER), PLANE_HEIGHT_PX, PLANE_WIDTH_PX)
    assert planes.dtype == np.uint16
    assert frames[0].frame_id == 1
    assert frames[2].frame_id == 3


def test_build_frames_renders_uint16_planes() -> None:
    """Rendered frames stay inside the 12-bit range."""
    frames = build_frames(num_frames=1, seed=7)
    planes = np.asarray(frames[0].planes)
    assert planes.dtype == np.uint16
    assert int(planes.max()) <= 4095


def test_build_frames_deterministic_for_seed() -> None:
    """The same seed renders identical frames."""
    a = np.asarray(build_frames(num_frames=1, seed=3)[0].planes)
    b = np.asarray(build_frames(num_frames=1, seed=3)[0].planes)
    np.testing.assert_array_equal(a, b)


def test_plume_brightens_red_near_the_plume() -> None:
    """The red plane is brighter at the plume than in a far corner."""
    frames = build_frames(num_frames=1, seed=0)
    red = np.asarray(frames[0].planes)[2]
    assert float(red[20:60, 1012:1052].mean()) > float(red[:40, :40].mean())


def test_plume_detector_finds_one_blob() -> None:
    """The scripted plume detector yields exactly one blob on a processed frame."""
    detector = plume_detector()
    frame = ProcessedFrameMsg(
        msg_type=MessageType.PROCESSED_FRAME,
        timestamp_utc="t",
        frame_id=1,
        tensor=np.zeros((len(BAND_ORDER), DETECTOR_HEIGHT_PX, DETECTOR_WIDTH_PX), dtype=np.float32),
        quality_flags=frozenset(),
    )
    result = detector.detect(frame)
    assert isinstance(result, Ok)
    assert len(result.value.blobs) == 1
