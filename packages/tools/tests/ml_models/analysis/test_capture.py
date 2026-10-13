"""Tests for the capture-sink protocol scaffold."""

from flight.libs.types import Ok, Result
from tools.ml_models.analysis.capture import CaptureSink


class _Sink:
    """Minimal structural CaptureSink implementation."""

    def close(self) -> Result[None, str]:
        return Ok(None)


class _NotSink:
    """Object without the close protocol method."""


def test_capture_sink_is_runtime_checkable() -> None:
    """isinstance honours the structural protocol."""
    assert isinstance(_Sink(), CaptureSink)
    assert not isinstance(_NotSink(), CaptureSink)


def test_capture_sink_close_returns_result() -> None:
    """A conforming sink returns a Result from close."""
    sink = _Sink()
    assert isinstance(sink.close(), Ok)
