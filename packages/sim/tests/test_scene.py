"""Tests for SIL plume scene generation."""

import numpy as np
from flight.libs.messages import ProcessedFrameMsg
from flight.libs.types import MessageType, Ok
from flight.payload.preprocess import separate_bands
from sim.scene import build_frames, plume_detector


def test_build_frames_count_and_shape() -> None:
    """build_frames returns N frames each a (1024, 1024) uint16 mosaic plane."""
    frames = build_frames(3)
    assert len(frames) == 3
    mosaic = np.asarray(frames[0].mosaic)
    assert mosaic.shape == (1024, 1024)
    assert mosaic.dtype == np.uint16
    assert frames[0].frame_id == 1
    assert frames[2].frame_id == 3


def test_build_frames_renders_uint16_mosaic() -> None:
    """Rendered frames are 1024x1024 uint16 mosaics within 12-bit range."""
    frames = build_frames(num_frames=3, seed=7)
    assert len(frames) == 3
    mosaic = np.asarray(frames[0].mosaic)
    assert mosaic.shape == (1024, 1024)
    assert mosaic.dtype == np.uint16
    assert int(mosaic.max()) <= 4095


def test_build_frames_deterministic_for_seed() -> None:
    """The same seed renders identical frames (SIL determinism)."""
    a = np.asarray(build_frames(num_frames=1, seed=3)[0].mosaic)
    b = np.asarray(build_frames(num_frames=1, seed=3)[0].mosaic)
    np.testing.assert_array_equal(a, b)


def test_plume_brightens_nir_at_center() -> None:
    """The NIR plane is brighter inside the plume region than the background."""
    frames = build_frames(num_frames=1, seed=0)
    planes = separate_bands(np.asarray(frames[0].mosaic, dtype=np.float32))
    assert isinstance(planes, Ok)
    nir = planes.value[3]  # layout (BLUE, GREEN, RED, NIR) -> NIR is plane 3
    assert float(nir[325:355, 325:355].mean()) > float(nir[:40, :40].mean())


def test_plume_detector_finds_one_blob() -> None:
    """The scripted plume detector yields exactly one blob on a processed frame."""
    detector = plume_detector()
    frame = ProcessedFrameMsg(
        msg_type=MessageType.PROCESSED_FRAME,
        timestamp_utc="t",
        frame_id=1,
        tensor=np.zeros((4, 256, 256), dtype=np.float32),
        quality_flags=frozenset(),
        crop_origin_px=(0, 0),
        scale_factor=1.0,
    )
    result = detector.detect(frame)
    assert isinstance(result, Ok)
    assert len(result.value.blobs) == 1
