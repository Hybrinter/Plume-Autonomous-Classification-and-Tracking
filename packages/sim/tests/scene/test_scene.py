"""Tests for SIL plume scene generation."""

import numpy as np
from flight.libs.messages import ProcessedFrameMsg
from flight.libs.types import MessageType, Ok
from sim.scene import build_frames, plume_detector
from sim.scene.plume import FRAME_HEIGHT_PX, FRAME_WIDTH_PX

_PLUME_X = FRAME_WIDTH_PX / 2.0
_PLUME_Y = FRAME_HEIGHT_PX / 2.0 + (124.0 - 512.0) * (FRAME_HEIGHT_PX / 1024.0)


def test_build_frames_count_and_shape() -> None:
    """build_frames returns N frames each a (3, 1544, 2064) uint16 prism buffer."""
    frames = build_frames(3)
    assert len(frames) == 3
    mosaic = np.asarray(frames[0].mosaic)
    assert mosaic.shape == (3, FRAME_HEIGHT_PX, FRAME_WIDTH_PX)
    assert mosaic.dtype == np.uint16
    assert frames[0].frame_id == 1
    assert frames[2].frame_id == 3


def test_build_frames_renders_uint16_buffer() -> None:
    """Rendered frames are 3x1544x2064 uint16 buffers within 12-bit range."""
    frames = build_frames(num_frames=3, seed=7)
    assert len(frames) == 3
    mosaic = np.asarray(frames[0].mosaic)
    assert mosaic.shape == (3, FRAME_HEIGHT_PX, FRAME_WIDTH_PX)
    assert mosaic.dtype == np.uint16
    assert int(mosaic.max()) <= 4095


def test_build_frames_deterministic_for_seed() -> None:
    """The same seed renders identical frames (SIL determinism)."""
    a = np.asarray(build_frames(num_frames=1, seed=3)[0].mosaic)
    b = np.asarray(build_frames(num_frames=1, seed=3)[0].mosaic)
    np.testing.assert_array_equal(a, b)


def test_plume_brightens_red_at_center() -> None:
    """The red plane is brighter inside the plume region than the background."""
    frames = build_frames(num_frames=1, seed=0)
    planes = np.asarray(frames[0].mosaic, dtype=np.float32)
    red = planes[0]
    y = int(round(_PLUME_Y))
    x = int(round(_PLUME_X))
    assert float(red[y - 20 : y + 20, x - 20 : x + 20].mean()) > float(red[:40, :40].mean())


def test_plume_detector_finds_one_blob() -> None:
    """The scripted plume detector yields exactly one blob on a processed frame."""
    detector = plume_detector()
    frame = ProcessedFrameMsg(
        msg_type=MessageType.PROCESSED_FRAME,
        timestamp_utc="t",
        frame_id=1,
        tensor=np.zeros((1, 3, FRAME_HEIGHT_PX, FRAME_WIDTH_PX), dtype=np.float32),
        quality_flags=frozenset(),
    )
    result = detector.detect(frame)
    assert isinstance(result, Ok)
    assert len(result.value.blobs) == 1
