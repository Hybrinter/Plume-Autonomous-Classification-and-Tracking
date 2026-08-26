"""Tests for tools.model metrics and scaffold modules."""

import pytest
from tools.model.export import export
from tools.model.metrics import binary_accuracy, mean_binary_accuracy
from tools.model.train import train


def test_binary_accuracy_match_and_mismatch() -> None:
    """binary_accuracy is 1.0 on a match and 0.0 on a mismatch."""
    assert binary_accuracy(True, True) == 1.0
    assert binary_accuracy(False, True) == 0.0


def test_mean_binary_accuracy() -> None:
    """mean_binary_accuracy averages pairwise scores."""
    assert mean_binary_accuracy((True, False), (True, False)) == 1.0
    assert mean_binary_accuracy((True, True), (True, False)) == 0.5
    assert mean_binary_accuracy((), ()) == 0.0


def test_mean_binary_accuracy_rejects_length_mismatch() -> None:
    """Unequal prediction/label lengths raise ValueError."""
    with pytest.raises(ValueError):
        mean_binary_accuracy((True,), (True, False))


def test_train_scaffold_raises() -> None:
    """train() is a stub in this layer."""
    with pytest.raises(NotImplementedError):
        train()


def test_export_scaffold_raises() -> None:
    """export() is a stub in this layer."""
    with pytest.raises(NotImplementedError):
        export()
