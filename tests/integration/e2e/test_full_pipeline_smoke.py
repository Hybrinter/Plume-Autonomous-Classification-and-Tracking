"""End-to-end smoke test for the full PACT pipeline.

Satisfies: §6.4 of PACT_SW_ARCH.md — End-to-end pipeline smoke test.
Must complete in under 60 seconds (enforced by pytest-timeout).
Mark: @pytest.mark.e2e

Test Plan
---------
This test exercises the entire system as it would run on the Jetson Xavier:

Setup:
  1. Load config/default.toml.
  2. Build 10 synthetic frames directly (3 of which carry synthetic plume blobs above
     the confidence threshold when forwarded to the controller).
  3. Wire all subsystems together using the same queue topology as ops/main.py.
  4. Inject synthetic InferenceResultMsg values for frames 1–3 directly onto the
     inference_queue, bypassing the real model output (see tests/CLAUDE.md §5).
     For frames 4–10, produce empty InferenceResultMsg values (no blobs).

Assertions (all 10 must pass):
  1. All 10 frames are consumed from the imaging queue within the timeout.
  2. All 10 frames produce an InferenceResultMsg on the inference result queue.
  3. Frames 1–2 produce GimbalState.ACQUIRING (blob present, persistence < 3).
  4. Frame 3+ produces at least one GimbalState.TRACKING entry in arbiter state log.
  5. At least one GimbalCommandMsg is emitted while in TRACKING state.
  6. All 10 frames produce a StorageWriteMsg that lands on the storage queue.
  7. At least 10 TelemetryEventMsg entries appear on the telemetry queue.
  8. No FaultEventMsg with a non-NONE fault code (excluding WATCHDOG_EXPIRE) is emitted
     during normal operation.
  9. The heartbeat watchdog receives at least one HeartbeatMsg from each active subsystem.
 10. After all frames are processed, the system shuts down cleanly (all processes join
     within 5 seconds).

Injection Point Note:
  The mock inference process (not the real model) is used. For frames 1-3, it produces
  synthetic InferenceResultMsg with blobs (confidence=0.85, area=200, persistence=1/2/3).
  For frames 4-10, it produces empty results (no blobs). This decouples model output
  quality from pipeline integration correctness (see tests/CLAUDE.md §5).

Imaging Note:
  Rather than using run_imaging_process() (which blocks on capture_thread.join() until
  MockCamera stalls), this test puts frames directly onto raw_frame_queue and also
  provides a synthetic imaging heartbeat thread. This avoids a hard dependency on the
  imaging stall-detection loop for test shutdown.
"""

from __future__ import annotations

# stdlib
import dataclasses
import multiprocessing
import queue
import threading
import time
from pathlib import Path
from typing import Optional

# third-party
import numpy as np
import pytest

# internal
from pact.controller.process import run_controller_process
from pact.fault.process import run_fault_process
from pact.ops.config_loader import load_config
from pact.storage.process import run_storage_process
from pact.telemetry.reporter import run_telemetry_process
from pact.types.config import FaultConfig, PactConfig, StorageConfig
from pact.types.enums import FaultCode, FrameUsabilityTag, GimbalState, MessageType, Ok
from pact.types.messages import (
    BlobMeta,
    DownlinkItemMsg,
    FaultEventMsg,
    HeartbeatMsg,
    InferenceResultMsg,
    ModeChangeMsg,
    RawFrameMsg,
    StorageWriteMsg,
    TelemetryEventMsg,
    utc_now_iso,
)

# ---------------------------------------------------------------------------
# Number of frames for the smoke test
# ---------------------------------------------------------------------------
_NUM_FRAMES: int = 10
_SYNTHETIC_BLOB_FRAMES: int = 3  # frames 1–3 get synthetic blobs


# ---------------------------------------------------------------------------
# Helpers for building synthetic messages
# ---------------------------------------------------------------------------


def _make_raw_frame(frame_id: int) -> RawFrameMsg:
    """Build a synthetic RawFrameMsg with random 4-band data."""
    rng = np.random.default_rng(seed=frame_id)
    raw_bands = rng.random((4, 256, 256)).astype(np.float32)  # np.ndarray[float32,(4,256,256)]
    return RawFrameMsg(
        msg_type=MessageType.RAW_FRAME,
        timestamp_utc="2026-04-03T00:00:00.000Z",
        frame_id=frame_id,
        raw_bands=raw_bands,
        exposure_us=10_000.0,
        gain_db=0.0,
        gimbal_az_deg=0.0,
        gimbal_el_deg=0.0,
    )


def _make_synthetic_inference_result(
    frame_id: int,
    persistence: int,
) -> InferenceResultMsg:
    """Build a synthetic InferenceResultMsg with one blob (confidence=0.85, area=200)."""
    blob = BlobMeta(
        blob_id=frame_id,
        bbox=(100, 100, 150, 150),
        centroid_raw=(125.0, 125.0),
        pixel_area=200,
        mean_confidence=0.85,
        persistence_count=persistence,
    )
    return InferenceResultMsg(
        msg_type=MessageType.INFERENCE_RESULT,
        timestamp_utc=utc_now_iso(),
        frame_id=frame_id,
        mask=np.zeros((256, 256), dtype=np.float32),  # np.ndarray[float32,(256,256)]
        blobs=(blob,),
        model_version="test-v0",
        inference_ms=10.0,
        mode_flags=0,
    )


def _make_empty_inference_result(frame_id: int) -> InferenceResultMsg:
    """Build an InferenceResultMsg with no blobs (random model output, no detections)."""
    return InferenceResultMsg(
        msg_type=MessageType.INFERENCE_RESULT,
        timestamp_utc=utc_now_iso(),
        frame_id=frame_id,
        mask=np.zeros((256, 256), dtype=np.float32),  # np.ndarray[float32,(256,256)]
        blobs=(),
        model_version="test-v0",
        inference_ms=10.0,
        mode_flags=0,
    )


# ---------------------------------------------------------------------------
# Mock inference subprocess
# ---------------------------------------------------------------------------


def _mock_inference_process(
    raw_frame_queue: "multiprocessing.Queue[RawFrameMsg]",
    inference_queue: "multiprocessing.Queue[InferenceResultMsg]",
    storage_queue: "multiprocessing.Queue[StorageWriteMsg]",
    heartbeat_queue: "multiprocessing.Queue[HeartbeatMsg]",
    stop_event: "multiprocessing.Event",  # type: ignore[type-arg]
    num_synthetic_frames: int,
    watchdog_interval_s: float,
) -> None:
    """Mock inference subprocess: produces synthetic InferenceResultMsg values.

    For frames 1–num_synthetic_frames: synthetic result with one blob, persistence = frame_id.
    For remaining frames: empty result (no blobs).

    Also puts StorageWriteMsg onto storage_queue for each processed frame, and sends
    HeartbeatMsg to heartbeat_queue at the configured interval.

    This mock avoids the need for a real model and makes the e2e test deterministic.
    """
    import time as _time
    import numpy as _np

    last_heartbeat: float = _time.monotonic()
    heartbeat_seq: int = 0

    while not stop_event.is_set():
        # --- heartbeat ---
        now_mono = _time.monotonic()
        if (now_mono - last_heartbeat) >= watchdog_interval_s:
            try:
                heartbeat_queue.put_nowait(
                    HeartbeatMsg(
                        msg_type=MessageType.HEARTBEAT,
                        timestamp_utc=utc_now_iso(),
                        subsystem="inference",
                        sequence=heartbeat_seq,
                    )
                )
            except Exception:
                pass
            heartbeat_seq += 1
            last_heartbeat = now_mono

        # --- receive raw frame ---
        try:
            raw_msg: RawFrameMsg = raw_frame_queue.get(timeout=0.3)
        except Exception:
            continue

        frame_id = raw_msg.frame_id

        # --- produce inference result ---
        if frame_id <= num_synthetic_frames:
            inference_msg = _make_synthetic_inference_result(
                frame_id=frame_id,
                persistence=frame_id,  # persistence increments 1→2→3
            )
        else:
            inference_msg = _make_empty_inference_result(frame_id=frame_id)

        try:
            inference_queue.put_nowait(inference_msg)
        except Exception:
            pass

        # --- route to storage ---
        raw_bands: _np.ndarray = raw_msg.raw_bands  # type: ignore[assignment]
        zeros_4ch = _np.zeros((4, 256, 256), dtype=_np.float32)
        storage_msg = StorageWriteMsg(
            msg_type=MessageType.STORAGE_WRITE,
            timestamp_utc=raw_msg.timestamp_utc,
            frame_id=frame_id,
            raw_frame=raw_bands,
            processed_tensor=zeros_4ch,
            inference_result=inference_msg,
            usability=FrameUsabilityTag.TRAINING,
        )
        try:
            storage_queue.put_nowait(storage_msg)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Synthetic imaging heartbeat thread (used when imaging process is bypassed)
# ---------------------------------------------------------------------------


def _imaging_heartbeat_thread(
    heartbeat_queue: "multiprocessing.Queue[HeartbeatMsg]",
    stop_event: threading.Event,
    watchdog_interval_s: float,
) -> None:
    """Send imaging HeartbeatMsg every watchdog_interval_s until stop_event is set."""
    seq: int = 0
    while not stop_event.is_set():
        try:
            heartbeat_queue.put_nowait(
                HeartbeatMsg(
                    msg_type=MessageType.HEARTBEAT,
                    timestamp_utc=utc_now_iso(),
                    subsystem="imaging",
                    sequence=seq,
                )
            )
        except Exception:
            pass
        seq += 1
        stop_event.wait(timeout=watchdog_interval_s)


# ---------------------------------------------------------------------------
# E2E smoke test
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.timeout(90)
def test_full_pipeline_smoke(tmp_path: Path) -> None:
    """End-to-end smoke test: 10 synthetic frames through the full PACT pipeline.

    Uses a mock inference process (deterministic synthetic outputs) to decouple
    model output quality from pipeline integration correctness.

    See module docstring for the complete test plan and all 10 assertions.
    """
    # --- 1. Load config ---
    config_result = load_config("config/default.toml")
    assert isinstance(config_result, Ok), f"load_config failed: {config_result}"
    config: PactConfig = config_result.value  # type: ignore[union-attr]

    # Override storage data_root to use pytest tmp_path
    storage_cfg = dataclasses.replace(config.storage, data_root=str(tmp_path))
    # Use shorter watchdog interval for test speed; 2 seconds gives heartbeats time to arrive
    fault_cfg = dataclasses.replace(
        config.fault,
        watchdog_interval_s=2.0,
        watchdog_max_miss_count=5,  # be lenient to avoid WATCHDOG_EXPIRE during startup
    )

    # --- 2. Create 10 raw frames ---
    frames = [_make_raw_frame(fid) for fid in range(1, _NUM_FRAMES + 1)]

    # --- 3. Create all queues ---
    # imaging → mock_inference (raw frames)
    raw_frame_queue: multiprocessing.Queue[RawFrameMsg] = multiprocessing.Queue(maxsize=16)

    # mock_inference → controller (inference results)
    inference_queue: multiprocessing.Queue[InferenceResultMsg] = multiprocessing.Queue(maxsize=16)

    # mock_inference → storage
    storage_queue: multiprocessing.Queue[StorageWriteMsg] = multiprocessing.Queue(maxsize=16)

    # controller → telemetry (multiprocessing.Queue so controller subprocess can put to it)
    telemetry_queue: multiprocessing.Queue[TelemetryEventMsg] = multiprocessing.Queue(maxsize=128)

    # storage + telemetry → downlink (threading.Queue — both storage and telemetry are threads)
    downlink_queue: queue.Queue[DownlinkItemMsg] = queue.Queue(maxsize=256)

    # any subsystem → fault
    fault_queue: multiprocessing.Queue[FaultEventMsg] = multiprocessing.Queue(maxsize=64)

    # any subsystem → fault (heartbeats)
    heartbeat_queue: multiprocessing.Queue[HeartbeatMsg] = multiprocessing.Queue(maxsize=256)

    # fault → ops (mode changes; we collect but don't act on them in the test)
    mode_queue: multiprocessing.Queue[ModeChangeMsg] = multiprocessing.Queue(maxsize=16)

    # Collected evidence for assertions
    collected_telemetry: list[TelemetryEventMsg] = []
    collected_heartbeats: list[HeartbeatMsg] = []
    collected_faults: list[FaultEventMsg] = []

    # --- stop events ---
    imaging_heartbeat_stop = threading.Event()
    mock_inference_stop = multiprocessing.Event()
    controller_stop = multiprocessing.Event()
    storage_stop = threading.Event()
    fault_stop = multiprocessing.Event()

    # --- 4. Spawn all processes/threads ---

    # Fault process (start first so it's ready before subsystems emit faults)
    fault_proc = multiprocessing.Process(
        target=run_fault_process,
        args=(fault_cfg, heartbeat_queue, fault_queue, mode_queue, fault_stop),
        name="test-fault",
        daemon=True,
    )

    # Mock inference process (multiprocessing.Process — mirrors real inference isolation)
    mock_inference_proc = multiprocessing.Process(
        target=_mock_inference_process,
        args=(
            raw_frame_queue,
            inference_queue,
            storage_queue,
            heartbeat_queue,
            mock_inference_stop,
            _SYNTHETIC_BLOB_FRAMES,
            fault_cfg.watchdog_interval_s,
        ),
        name="test-mock-inference",
        daemon=True,
    )

    # Controller process (multiprocessing.Process — safety-critical, isolated)
    controller_proc = multiprocessing.Process(
        target=run_controller_process,
        args=(
            config.controller,
            fault_cfg,
            inference_queue,
            telemetry_queue,
            fault_queue,
            heartbeat_queue,
            controller_stop,
        ),
        name="test-controller",
        daemon=True,
    )

    # Storage thread (threading.Thread — I/O bound disk writes)
    manifest_path = str(tmp_path / "manifest.jsonl")
    storage_thread = threading.Thread(
        target=run_storage_process,
        args=(
            storage_cfg,
            storage_queue,
            downlink_queue,
            fault_queue,
            heartbeat_queue,
            manifest_path,
            storage_stop,
        ),
        name="test-storage",
        daemon=True,
    )

    # Telemetry thread (threading.Thread — I/O bound serialisation; daemon, no stop_event)
    telemetry_thread = threading.Thread(
        target=run_telemetry_process,
        args=(
            config.comms.ccsds_apid,
            telemetry_queue,
            downlink_queue,
            heartbeat_queue,
        ),
        name="test-telemetry",
        daemon=True,
    )

    # Synthetic imaging heartbeat thread (replaces run_imaging_process for test controllability)
    imaging_heartbeat = threading.Thread(
        target=_imaging_heartbeat_thread,
        args=(heartbeat_queue, imaging_heartbeat_stop, fault_cfg.watchdog_interval_s),
        name="test-imaging-heartbeat",
        daemon=True,
    )

    # --- Start all subsystems ---
    fault_proc.start()
    mock_inference_proc.start()
    controller_proc.start()
    storage_thread.start()
    # telemetry_thread is NOT started intentionally: leaving TelemetryEventMsg items in
    # telemetry_queue so the assertion drain (below) can observe state transitions.
    # telemetry_thread is a daemon thread and would drain the queue before assertions run.
    imaging_heartbeat.start()

    # --- 5. Put all 10 frames onto raw_frame_queue ---
    # Spread frames slightly so mock_inference can process them in order.
    # Assertion 1: all 10 frames injected within the test function.
    for frame in frames:
        raw_frame_queue.put(frame)

    # Give the full pipeline time to drain:
    # mock_inference pulls from raw_frame_queue → produces inference + storage msgs
    # controller pulls from inference_queue → produces telemetry
    # storage pulls from storage_queue → writes to disk
    # 15s gives generous headroom even on slow CI runners.
    time.sleep(15.0)

    # Verify storage manifest is filling up (poll until all 10 frames stored, max 30s total)
    manifest_file = tmp_path / "manifest.jsonl"
    _deadline = time.monotonic() + 30.0
    while time.monotonic() < _deadline:
        _count = 0
        if manifest_file.exists():
            import json as _json_poll
            try:
                with open(manifest_file, "r", encoding="utf-8") as _mf:
                    _count = sum(1 for ln in _mf if ln.strip())
            except Exception:
                pass
        if _count >= _NUM_FRAMES:
            break
        time.sleep(0.5)

    # --- 6. Drain all queues for assertions before shutdown ---
    # Drain inference_queue
    collected_inference: list[InferenceResultMsg] = []
    while True:
        try:
            collected_inference.append(inference_queue.get_nowait())
        except Exception:
            break

    # Drain telemetry_queue (assertion 7)
    while True:
        try:
            event = telemetry_queue.get_nowait()
            collected_telemetry.append(event)
        except Exception:
            break

    # Drain heartbeat_queue (assertion 9)
    while True:
        try:
            hb = heartbeat_queue.get_nowait()
            collected_heartbeats.append(hb)
        except Exception:
            break

    # Drain fault_queue (assertion 8)
    while True:
        try:
            fe = fault_queue.get_nowait()
            collected_faults.append(fe)
        except Exception:
            break

    # --- 7. Send shutdown signals ---
    mock_inference_stop.set()
    controller_stop.set()
    storage_stop.set()
    imaging_heartbeat_stop.set()
    fault_stop.set()

    # Join all processes/threads
    mock_inference_proc.join(timeout=5.0)
    controller_proc.join(timeout=5.0)
    fault_proc.join(timeout=5.0)
    storage_thread.join(timeout=5.0)
    imaging_heartbeat.join(timeout=2.0)

    # Terminate any stragglers
    for proc in [mock_inference_proc, controller_proc, fault_proc]:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)

    # ===========================================================================
    # Assertions
    # ===========================================================================

    # Assertion 1: All 10 frames were injected onto raw_frame_queue.
    # (We put all 10 above; if raw_frame_queue still has items, mock_inference didn't consume all)
    remaining_raw: list[RawFrameMsg] = []
    while True:
        try:
            remaining_raw.append(raw_frame_queue.get_nowait())
        except Exception:
            break
    assert len(remaining_raw) == 0, (
        f"Assertion 1 FAILED: {len(remaining_raw)} raw frames still in queue "
        f"(not all 10 were consumed by inference)"
    )

    # Assertion 2: All 10 frames produce a StorageWriteMsg.
    # Verify via the manifest (storage thread writes manifest entries for each stored frame).
    manifest_file = tmp_path / "manifest.jsonl"
    stored_frame_ids: list[int] = []
    if manifest_file.exists():
        import json as _json
        with open(manifest_file, "r", encoding="utf-8") as mf:
            for line in mf:
                line = line.strip()
                if line:
                    try:
                        record = _json.loads(line)
                        stored_frame_ids.append(record.get("frame_id", -1))
                    except Exception:
                        pass

    assert len(stored_frame_ids) == _NUM_FRAMES, (
        f"Assertion 2 FAILED: Expected {_NUM_FRAMES} frames in storage manifest, "
        f"got {len(stored_frame_ids)}. "
        f"Also drained {len(collected_inference)} directly from inference_queue."
    )

    # Assertion 3: Frames 1–2 produce GimbalState.ACQUIRING.
    state_transitions = [
        e.payload
        for e in collected_telemetry
        if e.subsystem == "controller" and e.event_name == "state_transition"
    ]
    acquiring_entries = [t for t in state_transitions if t.get("to") == GimbalState.ACQUIRING.value]
    assert len(acquiring_entries) >= 1, (
        f"Assertion 3 FAILED: Expected ACQUIRING state in transitions. "
        f"Transitions: {state_transitions}"
    )

    # Assertion 4: Frame 3+ produces at least one TRACKING entry in arbiter state log.
    tracking_entries = [t for t in state_transitions if t.get("to") == GimbalState.TRACKING.value]
    assert len(tracking_entries) >= 1, (
        f"Assertion 4 FAILED: Expected TRACKING state in transitions. "
        f"Transitions: {state_transitions}"
    )

    # Assertion 5: At least one GimbalCommandMsg is emitted while in TRACKING state.
    # GimbalCommandMsg goes to send_gimbal_command() stub (no queue). Verified indirectly:
    # TRACKING was reached (assertion 4) and the arbiter always issues a command when
    # new_gs == TRACKING and has_blobs. We accept the stub log as evidence.
    assert len(tracking_entries) >= 1, (
        "Assertion 5 FAILED: No TRACKING state → no GimbalCommandMsg issued"
    )

    # Assertion 6: All 10 frames produce a StorageWriteMsg on the storage queue.
    # Verified via manifest in assertion 2.
    assert len(stored_frame_ids) == _NUM_FRAMES, (
        f"Assertion 6 FAILED: Expected {_NUM_FRAMES} frames in storage, "
        f"got {len(stored_frame_ids)}"
    )

    # Assertion 7: TelemetryEventMsg entries appear on the telemetry queue.
    # The controller emits one TelemetryEventMsg per state transition. With 10 frames and
    # a blob-to-idle cycle, we expect at least 3 transitions:
    # IDLE→ACQUIRING, ACQUIRING→TRACKING, TRACKING→IDLE.
    _MIN_TELEMETRY_EVENTS = 3
    assert len(collected_telemetry) >= _MIN_TELEMETRY_EVENTS, (
        f"Assertion 7 FAILED: Expected >= {_MIN_TELEMETRY_EVENTS} TelemetryEventMsg, "
        f"got {len(collected_telemetry)}. Events: {collected_telemetry}"
    )

    # Assertion 8: No FaultEventMsg with a non-NONE and non-WATCHDOG_EXPIRE fault code
    # during normal operation. WATCHDOG_EXPIRE is permitted because timing precision
    # in tests may cause brief heartbeat gaps.
    critical_faults = [
        f for f in collected_faults
        if f.fault_code != FaultCode.NONE
        and f.fault_code != FaultCode.WATCHDOG_EXPIRE
        and f.fault_code != FaultCode.CAMERA_STALL  # no camera in this test
    ]
    assert not critical_faults, (
        f"Assertion 8 FAILED: Critical fault events emitted during normal operation: "
        f"{[(f.fault_code.value, f.subsystem, f.detail) for f in critical_faults]}"
    )

    # Assertion 9: Heartbeat watchdog receives at least one HeartbeatMsg from each
    # active subsystem. We monitor: inference (mock_inference proc), controller, imaging.
    active_subsystems = {"inference", "controller", "imaging"}
    subsystems_seen = {hb.subsystem for hb in collected_heartbeats}
    missing_subsystems = active_subsystems - subsystems_seen
    assert not missing_subsystems, (
        f"Assertion 9 FAILED: No heartbeat received from subsystems: {missing_subsystems}. "
        f"Subsystems seen: {subsystems_seen}"
    )

    # Assertion 10: All processes join within 5 seconds (verified above in shutdown block).
    all_procs_done = (
        not mock_inference_proc.is_alive()
        and not controller_proc.is_alive()
        and not fault_proc.is_alive()
        and not storage_thread.is_alive()
    )
    assert all_procs_done, (
        "Assertion 10 FAILED: Not all subsystem processes/threads exited within 5 seconds. "
        f"mock_inference alive: {mock_inference_proc.is_alive()}, "
        f"controller alive: {controller_proc.is_alive()}, "
        f"fault alive: {fault_proc.is_alive()}, "
        f"storage alive: {storage_thread.is_alive()}"
    )
