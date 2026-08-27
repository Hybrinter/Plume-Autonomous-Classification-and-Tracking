"""Golden fixture tensors live in git under tests/fixtures/."""

from pathlib import Path

import numpy as np
from tools.model.accept import GoldenClassifierScene, GoldenScene, compute_iou

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_golden_fixtures_load() -> None:
    """Committed golden tensors have PACT rank and a planted blob."""
    positive = np.load(_FIXTURES / "golden_positive.npy")
    negative = np.load(_FIXTURES / "golden_negative.npy")
    mask = np.load(_FIXTURES / "golden_mask.npy")
    assert positive.shape == (4, 8, 8)
    assert negative.shape == (4, 8, 8)
    assert mask.shape == (8, 8)
    scene = GoldenScene(input_tensor=positive, gold_mask=mask)
    assert compute_iou(scene.gold_mask, mask) == 1.0
    clf = GoldenClassifierScene(input_tensor=positive, label_positive=True)
    assert clf.label_positive is True
    assert float(negative.sum()) == 0.0
