"""Tests for the Result/Ok/Err types."""

from flight.libs.types import Err, Ok


def test_ok_carries_value() -> None:
    """Ok stores its value and is matchable."""
    result = Ok(42)
    assert isinstance(result, Ok)
    assert result.value == 42


def test_err_carries_error() -> None:
    """Err stores its error and is distinguishable from Ok."""
    result = Err("boom")
    assert isinstance(result, Err)
    assert not isinstance(result, Ok)
    assert result.error == "boom"
