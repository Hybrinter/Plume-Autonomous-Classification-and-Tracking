"""Tests for the model GSD encoding."""

import numpy as np
import pytest
from flight.libs.types import Ok
from tools.ml_models.dataset import gsd
from tools.ml_models.dataset.gsd import to_model_gsd


def test_reference_gsd_encodes_to_zero() -> None:
    """GSD equal to the reference is the zero vector."""
    encoded = to_model_gsd(np.array([15.87, 15.87], dtype=np.float32), 15.87)
    np.testing.assert_allclose(encoded, np.zeros(2, dtype=np.float32), atol=1e-6)


def test_malformed_gsd_shape_is_rejected() -> None:
    """GSD input must carry a length-2 component axis."""
    with pytest.raises(ValueError, match="flight GSD encoding"):
        to_model_gsd(np.array([15.87, 15.87, 15.87], dtype=np.float32), 15.87)
    with pytest.raises(ValueError, match="flight GSD encoding"):
        to_model_gsd(np.zeros((2, 3), dtype=np.float32), 15.87)
    with pytest.raises(ValueError, match="flight GSD encoding"):
        to_model_gsd(np.empty((0, 2), dtype=np.float32), 15.87)


def test_gsd_wrapper_delegates_reference_and_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public dataset helper unwraps the flight GSD result contract."""
    values = np.array([15.87, 15.87], dtype=np.float32)
    seen: list[tuple[np.ndarray, float]] = []

    def fake(values_arg: np.ndarray, reference_m: float) -> Ok[np.ndarray]:
        seen.append((values_arg, reference_m))
        return Ok(np.zeros_like(values_arg, dtype=np.float32))

    monkeypatch.setattr(gsd, "_flight_to_model_gsd", fake)
    np.testing.assert_array_equal(to_model_gsd(values, 15.87), np.zeros(2, dtype=np.float32))
    assert seen == [(values, 15.87)]
