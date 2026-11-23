"""Tile orchestration gates segmentation and preserves image/GSD alignment."""

import numpy as np
from flight.libs.messages import ProcessedFrameMsg
from flight.libs.types import Err, FaultCode, MessageType, Ok, Result
from flight.payload.inference import (
    Detector,
    ScriptedClassifier,
    ScriptedDetector,
    infer_tiles,
)


class FakeClassifier:
    def __init__(self, logits: np.ndarray | None = None) -> None:
        self.logits = logits
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def classify_tiles(self, images: np.ndarray, gsd: np.ndarray) -> Result[np.ndarray, FaultCode]:
        self.calls.append((images.copy(), gsd.copy()))
        if self.logits is None:
            return Ok(np.full(images.shape[0], -1.0, dtype=np.float32))
        return Ok(self.logits.copy())


class FakeSegmentor:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def segment_tiles(self, images: np.ndarray, gsd: np.ndarray) -> Result[np.ndarray, FaultCode]:
        self.calls.append((images.copy(), gsd.copy()))
        return Ok(np.ones((images.shape[0], 1, images.shape[2], images.shape[3]), dtype=np.float32))


def _frame(
    *,
    grid: tuple[int, int] = (2, 2),
    gsd: np.ndarray | None = None,
    tensor: np.ndarray | None = None,
) -> ProcessedFrameMsg:
    if tensor is None:
        tensor = np.arange(1 * 1 * 8 * 8, dtype=np.float32).reshape(1, 1, 8, 8)
    if gsd is None:
        gsd = np.full((grid[0] * grid[1], 2), 15.87, dtype=np.float32)
    return ProcessedFrameMsg(
        msg_type=MessageType.PROCESSED_FRAME,
        timestamp_utc="2026-10-01T00:00:00.000Z",
        frame_id=4,
        tensor=tensor,
        quality_flags=frozenset(),
        tile_gsd_m=gsd,
    )


def test_infer_tiles_segments_only_positive_tiles_and_scatters_masks() -> None:
    images = np.arange(4 * 1 * 2 * 2, dtype=np.float32).reshape(4, 1, 2, 2)
    gsd = np.zeros((4, 2), dtype=np.float32)
    classifier = FakeClassifier(np.array([-1, 1, -2, 2], dtype=np.float32))
    segmentor = FakeSegmentor()

    result = infer_tiles(classifier, segmentor, images, gsd)

    assert isinstance(result, Ok)
    assert len(classifier.calls) == 1 and classifier.calls[0][0].shape[0] == 4
    assert len(segmentor.calls) == 1
    np.testing.assert_array_equal(segmentor.calls[0][0], images[[1, 3]])
    np.testing.assert_array_equal(result.value.positive, [False, True, False, True])
    assert result.value.masks.shape == (4, 1, 2, 2)
    assert float(result.value.masks[[0, 2]].max()) == 0.0
    assert float(result.value.masks[[1, 3]].min()) == 1.0


def test_infer_tiles_skips_segmentor_when_all_negative() -> None:
    images = np.ones((3, 2, 4, 4), dtype=np.float32)
    classifier = FakeClassifier()
    segmentor = FakeSegmentor()

    result = infer_tiles(classifier, segmentor, images, np.zeros((3, 2), dtype=np.float32))

    assert isinstance(result, Ok)
    assert segmentor.calls == []
    assert not bool(result.value.positive.any())
    assert not bool(result.value.masks.any())


def test_detector_passes_encoded_per_tile_gsd_and_stitches_masks() -> None:
    classifier = FakeClassifier(np.array([-1, 1, -1, 1], dtype=np.float32))
    segmentor = FakeSegmentor()
    detector = Detector(
        classifier,
        segmentor,
        grid=(2, 2),
        min_blob_area_px=99,
        record_wall_clock=False,
    )
    actual_gsd = np.array([[10, 11], [20, 21], [30, 31], [40, 41]], dtype=np.float32)

    result = detector.detect(_frame(gsd=actual_gsd))

    assert isinstance(result, Ok)
    expected_encoded = np.log(actual_gsd[[1, 3]] / 15.87).astype(np.float32)
    np.testing.assert_allclose(segmentor.calls[0][1], expected_encoded, atol=1e-7)
    assert result.value.tile_logits == (-1.0, 1.0, -1.0, 1.0)
    assert result.value.tile_positive == (False, True, False, True)
    assert result.value.tile_gsd_m == tuple(tuple(float(x) for x in row) for row in actual_gsd)
    expected_mask = np.zeros((8, 8), dtype=np.float32)
    expected_mask[:4, 4:] = 1
    expected_mask[4:, 4:] = 1
    np.testing.assert_array_equal(result.value.mask, expected_mask)


def test_detector_rejects_missing_or_malformed_physical_gsd() -> None:
    detector = Detector(FakeClassifier(), FakeSegmentor(), grid=(2, 2), record_wall_clock=False)
    missing = _frame()
    missing = ProcessedFrameMsg(
        msg_type=missing.msg_type,
        timestamp_utc=missing.timestamp_utc,
        frame_id=missing.frame_id,
        tensor=missing.tensor,
        quality_flags=missing.quality_flags,
    )
    bad = detector.detect(missing)
    assert isinstance(bad, Err) and bad.error is FaultCode.FRAME_MALFORMED

    malformed = detector.detect(_frame(gsd=np.ones((3, 2), dtype=np.float32)))
    assert isinstance(malformed, Err) and malformed.error is FaultCode.FRAME_MALFORMED


def test_detector_rejects_nonfinite_model_logits_and_masks() -> None:
    images = np.ones((1, 1, 2, 2), dtype=np.float32)
    bad_logits = infer_tiles(
        FakeClassifier(np.array([np.nan], dtype=np.float32)),
        FakeSegmentor(),
        images,
        np.zeros((1, 2), dtype=np.float32),
    )
    assert isinstance(bad_logits, Err) and bad_logits.error is FaultCode.INFERENCE_NAN

    class BadMask:
        def segment_tiles(
            self, images: np.ndarray, gsd: np.ndarray
        ) -> Result[np.ndarray, FaultCode]:
            return Ok(np.full((images.shape[0], 1, *images.shape[2:]), np.nan, dtype=np.float32))

    bad_masks = infer_tiles(
        FakeClassifier(np.array([1], dtype=np.float32)),
        BadMask(),
        images,
        np.zeros((1, 2), dtype=np.float32),
    )
    assert isinstance(bad_masks, Err) and bad_masks.error is FaultCode.INFERENCE_NAN


def test_scripted_detector_keeps_small_no_gsd_fixture_convenience() -> None:
    mask = np.ones((8, 8), dtype=np.float32)
    frame = _frame()
    frame = ProcessedFrameMsg(
        msg_type=frame.msg_type,
        timestamp_utc=frame.timestamp_utc,
        frame_id=frame.frame_id,
        tensor=frame.tensor,
        quality_flags=frame.quality_flags,
    )
    result = ScriptedDetector(mask, grid=(1, 1), min_blob_area_px=100).detect(frame)
    assert isinstance(result, Ok)
    assert len(result.value.tile_gsd_m) == 1


def test_scripted_detector_eight_by_eight_grid_stitches_full_frame_mask() -> None:
    mask = np.zeros((16, 24), dtype=np.float32)
    mask[2:14, 3:21] = 0.9
    detector = ScriptedDetector(mask, grid=(8, 8), min_blob_area_px=4, confidence_gate=0.5)
    frame = _frame(
        grid=(8, 8),
        tensor=np.zeros((1, 1, 16, 24), dtype=np.float32),
        gsd=np.full((64, 2), 15.87, dtype=np.float32),
    )

    result = detector.detect(frame)

    assert isinstance(result, Ok)
    np.testing.assert_array_equal(result.value.mask, mask)
    assert len(result.value.tile_logits) == 64
    assert len(result.value.blobs) == 1


def test_scripted_negative_zero_logit_is_gated_negative() -> None:
    classifier = ScriptedClassifier(positive=False, logit=0.0)
    result = classifier.classify_tiles(
        np.ones((1, 1, 2, 2), dtype=np.float32), np.zeros((1, 2), dtype=np.float32)
    )
    assert isinstance(result, Ok)
    assert result.value[0] < 0
