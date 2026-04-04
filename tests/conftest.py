"""Shared pytest fixtures for the PACT test suite.

All fixtures are defined here and injected automatically by pytest. Do not redefine
these in individual test files. See tests/CLAUDE.md §3 for the full fixture catalogue.

Satisfies: §6.1 of PACT_SW_ARCH.md
"""

from __future__ import annotations

# third-party
import numpy as np
import pytest

# internal — types
from pact.types.enums import GimbalState, MessageType
from pact.types.enums import Ok  # Ok/Err defined in enums, re-exported via pact.types
from pact.types.messages import (
    BlobMeta,
    InferenceResultMsg,
    ProcessedFrameMsg,
    RawFrameMsg,
)

# ArbiterState lives in controller.arbiter
from pact.controller.arbiter import ArbiterState

# config loader and config types
from pact.ops.config_loader import load_config  # type: ignore[import]
from pact.types.config import PactConfig  # type: ignore[import]

# MockCamera lives in imaging
from pact.imaging.camera import MockCamera  # type: ignore[import]


# ---------------------------------------------------------------------------
# mock_camera
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_camera() -> MockCamera:
    """Return a MockCamera pre-loaded with 5 synthetic 4-band (4,256,256) float32 frames.

    Each frame contains uniform-random pixel values in [0, 1]. No blobs are injected.
    Use MockCamera(frames=[...]) directly in your test for custom frame content.
    """
    rng = np.random.default_rng(seed=0)
    frames = []
    for frame_id in range(1, 6):
        raw_bands = rng.random((4, 256, 256)).astype(np.float32)  # np.ndarray[float32,(4,256,256)]
        msg = RawFrameMsg(
            msg_type=MessageType.RAW_FRAME,
            timestamp_utc="2026-04-03T00:00:00.000Z",
            frame_id=frame_id,
            raw_bands=raw_bands,
            exposure_us=10_000.0,
            gain_db=0.0,
            gimbal_az_deg=0.0,
            gimbal_el_deg=0.0,
        )
        frames.append(msg)
    return MockCamera(frames=frames)


# ---------------------------------------------------------------------------
# sample_raw_frame_msg
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_raw_frame_msg() -> RawFrameMsg:
    """Return a RawFrameMsg with a zeros (4,256,256) float32 array and frame_id=1.

    Timestamp is fixed at 2026-04-03T00:00:00.000Z for determinism.
    """
    return RawFrameMsg(
        msg_type=MessageType.RAW_FRAME,
        timestamp_utc="2026-04-03T00:00:00.000Z",
        frame_id=1,
        raw_bands=np.zeros((4, 256, 256), dtype=np.float32),  # np.ndarray[float32,(4,256,256)]
        exposure_us=10_000.0,
        gain_db=0.0,
        gimbal_az_deg=0.0,
        gimbal_el_deg=0.0,
    )


# ---------------------------------------------------------------------------
# sample_processed_frame_msg
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_processed_frame_msg() -> ProcessedFrameMsg:
    """Return a ProcessedFrameMsg with a zeros (4,256,256) float32 tensor and no quality flags.

    crop_origin_px=(0,0), scale_factor=1.0 — no crop applied.
    """
    return ProcessedFrameMsg(
        msg_type=MessageType.PROCESSED_FRAME,
        timestamp_utc="2026-04-03T00:00:00.000Z",
        frame_id=1,
        tensor=np.zeros((4, 256, 256), dtype=np.float32),  # np.ndarray[float32,(4,256,256)]
        quality_flags=frozenset(),
        crop_origin_px=(0, 0),
        scale_factor=1.0,
    )


# ---------------------------------------------------------------------------
# sample_inference_result
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_inference_result() -> InferenceResultMsg:
    """Return an InferenceResultMsg with one blob at confidence=0.85, area=200, persistence=1.

    The single BlobMeta has blob_id=1, bbox=(100,100,150,150), centroid=(125.0,125.0).
    Use this fixture as a baseline; construct InferenceResultMsg directly if you need
    different blob counts, confidence, area, or persistence.
    """
    blob = BlobMeta(
        blob_id=1,
        bbox=(100, 100, 150, 150),
        centroid_raw=(125.0, 125.0),
        pixel_area=200,
        mean_confidence=0.85,
        persistence_count=1,
    )
    return InferenceResultMsg(
        msg_type=MessageType.INFERENCE_RESULT,
        timestamp_utc="2026-04-03T00:00:00.000Z",
        frame_id=1,
        mask=np.zeros((256, 256), dtype=np.float32),  # np.ndarray[float32,(256,256)]
        blobs=(blob,),
        model_version="test-v0",
        inference_ms=50.0,
        mode_flags=0,
    )


# ---------------------------------------------------------------------------
# default_config
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_config() -> PactConfig:
    """Return a PactConfig loaded from config/default.toml via load_config().

    Do NOT mutate this fixture. It is a frozen dataclass. Use dataclasses.replace()
    to derive a modified config for tests that need non-default thresholds.
    """
    result = load_config("config/default.toml")
    assert isinstance(result, Ok), f"load_config() returned an error: {result}"
    return result.value


# ---------------------------------------------------------------------------
# arbiter_idle_state
# ---------------------------------------------------------------------------


@pytest.fixture()
def arbiter_idle_state() -> ArbiterState:
    """Return an ArbiterState in GimbalState.IDLE with no tracked blobs.

    Use as the starting state for all arbiter state machine tests.
    """
    return ArbiterState(
        gimbal_state=GimbalState.IDLE,
        tracked_blobs=(),
        idle_duration_s=0.0,
        last_command_time=0.0,
        current_target_id=None,
    )
