"""Verifies env-driven driver selection: all-sim, missing sim_inputs, link=real."""

import dataclasses
import hashlib
import socket
import threading
from pathlib import Path

import flight.payload.inference.runtime as inference_runtime
import pytest
from flight.core.select_drivers import SimDriverInputs, select_drivers
from flight.hal.drivers_real import RealStationLink
from flight.hal.drivers_sim import (
    SimGimbal,
    SimIssEphemeris,
    SimScalarSensor,
    SimSensor,
    SimStationLink,
)
from flight.libs.config import PactConfig
from flight.libs.time import ManualClock
from flight.libs.types import Err, FaultCode, Ok
from flight.payload.inference import (
    InferenceRuntime,
    OnnxRuntimeFactory,
    RuntimeSession,
    ScriptedDetector,
)
from sim.scene import build_frames, plume_detector


def _all_sim_config() -> PactConfig:
    """A PactConfig with every driver axis forced to 'sim'."""
    base = PactConfig()
    drivers = dataclasses.replace(
        base.drivers,
        sensor="sim",
        gimbal="sim",
        compute="sim",
        link="sim",
        clock="sim",
        ephemeris="sim",
    )
    return dataclasses.replace(base, drivers=drivers)


def _sim_inputs() -> SimDriverInputs:
    """A populated SimDriverInputs for the all-sim path."""
    return SimDriverInputs(
        frames=build_frames(2),
        detector=plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )


def test_all_sim_returns_sim_drivers_and_passed_detector() -> None:
    """All-sim selection wires every sim driver and reuses the passed detector."""
    inputs = _sim_inputs()
    drivers = select_drivers(_all_sim_config(), ManualClock(), inputs)
    assert isinstance(drivers.sensor, SimSensor)
    assert isinstance(drivers.gimbal, SimGimbal)
    assert isinstance(drivers.ephemeris, SimIssEphemeris)
    assert isinstance(drivers.station, SimStationLink)
    assert isinstance(drivers.thermal_sensor, SimScalarSensor)
    assert isinstance(drivers.power_sensor, SimScalarSensor)
    session = drivers.inference.snapshot()
    assert session is not None
    assert session.identity == "scripted"
    assert session.backend is inputs.detector
    assert isinstance(session.backend, ScriptedDetector)


def test_sim_axis_without_inputs_raises() -> None:
    """A sim axis with sim_inputs=None is a programming error -> ValueError."""
    with pytest.raises(ValueError, match="sim_inputs"):
        select_drivers(_all_sim_config(), ManualClock(), None)


def test_real_compute_starts_empty_with_lazy_factory() -> None:
    """compute='real' constructs zero sessions; load() is what builds one."""
    config = _all_sim_config()
    config = dataclasses.replace(
        config,
        drivers=dataclasses.replace(config.drivers, compute="real"),
    )
    drivers = select_drivers(config, ManualClock(), _sim_inputs())
    assert isinstance(drivers.inference, InferenceRuntime)
    assert drivers.inference.snapshot() is None
    assert drivers.inference.identity == ""
    assert isinstance(drivers.inference.factory, OnnxRuntimeFactory)


def test_real_compute_load_returns_result_for_missing_models() -> None:
    """Unreadable model artifacts fail typed (MODEL_CORRUPT), not by raising."""
    config = _all_sim_config()
    config = dataclasses.replace(
        config,
        drivers=dataclasses.replace(config.drivers, compute="real"),
        inference=dataclasses.replace(
            config.inference,
            classifier_model_path="nonexistent-classifier.onnx",
            segmentor_model_path="nonexistent-segmentor.onnx",
        ),
    )
    drivers = select_drivers(config, ManualClock(), _sim_inputs())
    factory = drivers.inference.factory
    assert factory is not None
    loaded = factory.load(threading.Event())
    assert isinstance(loaded, Err)
    assert loaded.error is FaultCode.MODEL_CORRUPT
    assert drivers.inference.snapshot() is None


def test_real_compute_load_configures_tiled_dynamic_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load() resolves paths, hashes both files, and passes observed digests.

    The session identity is the ordered classifier/segmentor digest pair, a
    fingerprint of the files actually read -- not trusted manifest
    authenticity.
    """
    config = _all_sim_config()
    classifier_path = tmp_path / "classifier.onnx"
    segmentor_path = tmp_path / "segmentor.onnx"
    classifier_path.write_bytes(b"classifier-bytes")
    segmentor_path.write_bytes(b"segmentor-bytes-different")
    config = dataclasses.replace(
        config,
        drivers=dataclasses.replace(config.drivers, compute="real"),
        inference=dataclasses.replace(
            config.inference,
            tile_rows=4,
            tile_cols=8,
            use_int8=False,
            classifier_model_path=str(classifier_path),
            segmentor_model_path=str(segmentor_path),
        ),
    )
    captured: dict[str, object] = {}
    detector = _sim_inputs().detector

    def _capture_detector(**kwargs: object) -> ScriptedDetector:
        captured.update(kwargs)
        return detector

    monkeypatch.setattr(inference_runtime, "OnnxDetector", _capture_detector)
    drivers = select_drivers(config, ManualClock(), _sim_inputs())
    factory = drivers.inference.factory
    assert factory is not None
    loaded = factory.load(threading.Event())
    assert isinstance(loaded, Ok)
    session = loaded.value
    assert isinstance(session, RuntimeSession)
    assert session.backend is detector

    classifier_sha = hashlib.sha256(b"classifier-bytes").hexdigest()
    segmentor_sha = hashlib.sha256(b"segmentor-bytes-different").hexdigest()
    assert session.identity == f"onnx:{classifier_sha}:{segmentor_sha}"
    assert captured["classifier_sha256"] == classifier_sha
    assert captured["segmentor_sha256"] == segmentor_sha
    tile_height = config.inference.input_height_px // config.inference.tile_rows
    tile_width = config.inference.input_width_px // config.inference.tile_cols
    assert captured["grid"] == (4, 8)
    assert captured["gsd_reference_m"] == config.inference.gsd_reference_m
    assert captured["confidence_gate"] == config.controller.vision.confidence_gate
    assert captured["min_blob_area_px"] == config.controller.vision.min_blob_area_px
    assert captured["logit_threshold"] == config.inference.classifier_logit_threshold
    assert captured["latency_budget_ms"] == config.fault.inference_timeout_ms
    assert captured["expected_input_shape"] == (None, 3, tile_height, tile_width)
    assert captured["expected_gsd_shape"] == (None, 2)
    assert captured["expected_classifier_output_shape"] == (None, 1)
    assert captured["expected_segmentor_output_shape"] == (None, 1, tile_height, tile_width)


@pytest.mark.parametrize(
    "sdk_error",
    [
        RuntimeError("onnxruntime exploded"),
        ImportError("onnxruntime is not installed"),
        ValueError("input shape contract mismatch"),
    ],
    ids=["runtime", "import", "contract"],
)
def test_real_compute_load_converts_sdk_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sdk_error: Exception
) -> None:
    """An SDK/contract failure inside OnnxDetector surfaces as Err, never raises."""
    config = _all_sim_config()
    for name in ("classifier.onnx", "segmentor.onnx"):
        (tmp_path / name).write_bytes(b"model")
    config = dataclasses.replace(
        config,
        drivers=dataclasses.replace(config.drivers, compute="real"),
        inference=dataclasses.replace(
            config.inference,
            use_int8=False,
            classifier_model_path=str(tmp_path / "classifier.onnx"),
            segmentor_model_path=str(tmp_path / "segmentor.onnx"),
        ),
    )

    def _failing_detector(**kwargs: object) -> ScriptedDetector:
        raise sdk_error

    monkeypatch.setattr(inference_runtime, "OnnxDetector", _failing_detector)
    drivers = select_drivers(config, ManualClock(), _sim_inputs())
    factory = drivers.inference.factory
    assert factory is not None
    loaded = factory.load(threading.Event())
    assert isinstance(loaded, Err)
    assert loaded.error is FaultCode.MODEL_CORRUPT


def test_real_compute_load_honors_cancellation(tmp_path: Path) -> None:
    """A pre-set cancel flag aborts the load with the timeout code."""
    config = _all_sim_config()
    for name in ("classifier.onnx", "segmentor.onnx"):
        (tmp_path / name).write_bytes(b"model")
    config = dataclasses.replace(
        config,
        drivers=dataclasses.replace(config.drivers, compute="real"),
        inference=dataclasses.replace(
            config.inference,
            use_int8=False,
            classifier_model_path=str(tmp_path / "classifier.onnx"),
            segmentor_model_path=str(tmp_path / "segmentor.onnx"),
        ),
    )
    drivers = select_drivers(config, ManualClock(), _sim_inputs())
    factory = drivers.inference.factory
    assert factory is not None
    cancel = threading.Event()
    cancel.set()
    loaded = factory.load(cancel)
    assert isinstance(loaded, Err)
    assert loaded.error is FaultCode.INFERENCE_TIMEOUT


def test_link_real_builds_realstationlink() -> None:
    """link='real' (others sim) builds a RealStationLink bound to a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]

    base = _all_sim_config()
    drivers_cfg = dataclasses.replace(base.drivers, link="real")
    link_cfg = dataclasses.replace(base.link, command_tcp_port=free_port)
    config = dataclasses.replace(base, drivers=drivers_cfg, link=link_cfg)

    drivers = select_drivers(config, ManualClock(), _sim_inputs())
    try:
        assert isinstance(drivers.station, RealStationLink)
        assert isinstance(drivers.sensor, SimSensor)
    finally:
        drivers.station.close()
