"""PACT top-level entry point.

Loads config, creates all inter-process queues, spawns all subsystem processes, and
runs the mode management loop.

Queue topology (all queues created here and passed as arguments):

    imaging_process    --[raw_frame_queue]-->   inference_process
                                                  (preprocessing runs inside inference process
                                                   on the hot path -- see preprocessing/adr/ADR-001)

    inference_process  --[inference_queue]-->   controller_process
    inference_process  --[storage_queue]-->     storage_process

    controller_process --[gimbal_queue]-->      (hardware gimbal driver, stub)
    controller_process --[telemetry_queue]-->   telemetry_process

    telemetry_process  --[downlink_queue]-->    comms_process
    storage_process    --[downlink_queue]-->    comms_process   (shared queue)

    comms_process      --[uplink_queue]-->      ops main (model deployment routing)

    any subsystem      --[fault_queue]-->       fault_process
    any subsystem      --[heartbeat_queue]-->   fault_process

    fault_process      --[mode_queue]-->        ops main (mode FSM)

Note: downlink_queue is shared between storage_process and telemetry_process.
      Both put DownlinkItemMsg onto it; comms_process drains it.

Satisfies: REQ-OPER-HIGH-002 (process orchestration and mode management).
"""

from __future__ import annotations

# stdlib
import dataclasses
import multiprocessing
import queue
import signal
import sys
import threading
import time
from typing import Optional

# third-party
import numpy as np
import structlog

# internal
from pact.comms.process import run_comms_process
from pact.comms.uplink import activate_staged_model, rollback_model, ModelUploadSession, process_uplink_chunk
from pact.controller.process import run_controller_process
from pact.fault.process import run_fault_process
from pact.imaging.process import run_imaging_process
from pact.model.architecture import build_model
from pact.model.inference import InferenceEngine
from pact.ops.config_loader import load_config
from pact.ops.modes import transition_mode
from pact.preprocessing.band_select import select_bands
from pact.preprocessing.quality import compute_quality_flags
from pact.preprocessing.radiometric import RadiometricCalibration, apply_calibration
from pact.storage.process import run_storage_process
from pact.telemetry.reporter import run_telemetry_process
from pact.types.config import PactConfig
from pact.types.enums import FaultCode, FrameUsabilityTag, MessageType, ModelDeployState, Ok, SystemMode
from pact.types.messages import (
    DownlinkItemMsg,
    FaultEventMsg,
    GimbalCommandMsg,
    HeartbeatMsg,
    InferenceResultMsg,
    ModeChangeMsg,
    ProcessedFrameMsg,
    RawFrameMsg,
    StorageWriteMsg,
    UploadChunkMsg,
    utc_now_iso,
)

log = structlog.get_logger().bind(subsystem="ops")


# ---------------------------------------------------------------------------
# Inference process entry point (preprocessing + inference on hot path)
# ---------------------------------------------------------------------------


def _run_inference_process(
    config: PactConfig,
    raw_frame_queue: "multiprocessing.Queue[RawFrameMsg]",
    inference_queue: "multiprocessing.Queue[InferenceResultMsg]",
    storage_queue: "multiprocessing.Queue[StorageWriteMsg]",
    fault_queue: "multiprocessing.Queue[FaultEventMsg]",
    heartbeat_queue: "multiprocessing.Queue[HeartbeatMsg]",
    stop_event: multiprocessing.Event,  # type: ignore[type-arg]
) -> None:
    """Inference process: preprocessing + model inference on the hot path.

    Runs as a multiprocessing.Process (REQ-AIML-COMP-002 — isolated from all other tasks).
    Preprocessing (band selection, radiometric correction, quality flags) runs here in-process
    to avoid serialisation overhead on RawFrameMsg → ProcessedFrameMsg.

    Satisfies: REQ-AIML-COMP-001, REQ-AIML-COMP-002, REQ-AIML-IMAG-002.
    """
    _log = structlog.get_logger().bind(subsystem="inference")
    _log.info("inference_process_start")

    # Build model and inference engine
    model = build_model(
        encoder_weights=None,  # use saved weights from model_path
        in_channels=len(config.inference.input_bands),
    )
    engine = InferenceEngine(
        model=model,
        config=config.inference,
        confidence_gate=config.controller.confidence_gate,
        min_blob_area_px=config.controller.min_blob_area_px,
    )

    # Dummy radiometric calibration (identity: zero dark frame, unit flat field)
    # Replace with real calibration matrices when sensor characterisation is complete.
    _H, _W = config.inference.input_height_px, config.inference.input_width_px
    _calib = RadiometricCalibration(
        dark_frame=np.zeros((4, _H, _W), dtype=np.float32),
        flat_field=np.ones((4, _H, _W), dtype=np.float32),
    )

    last_heartbeat: float = time.monotonic()
    heartbeat_seq: int = 0
    frame_seq: int = 0

    while not stop_event.is_set():
        # --- Heartbeat ---
        now_mono = time.monotonic()
        if (now_mono - last_heartbeat) >= config.fault.watchdog_interval_s:
            heartbeat_queue.put(
                HeartbeatMsg(
                    msg_type=MessageType.HEARTBEAT,
                    timestamp_utc=utc_now_iso(),
                    subsystem="inference",
                    sequence=heartbeat_seq,
                )
            )
            heartbeat_seq += 1
            last_heartbeat = now_mono

        # --- Receive raw frame ---
        try:
            raw_msg: RawFrameMsg = raw_frame_queue.get(timeout=1.0)
        except Exception:
            continue

        # --- Preprocessing (hot path — no queue round-trip) ---
        raw_bands: np.ndarray = raw_msg.raw_bands  # type: ignore[assignment]

        # Radiometric correction
        calib_result = apply_calibration(raw_bands, _calib)
        if not isinstance(calib_result, Ok):
            fault_queue.put(
                FaultEventMsg(
                    msg_type=MessageType.FAULT_EVENT,
                    timestamp_utc=utc_now_iso(),
                    fault_code=FaultCode.INFERENCE_NAN,
                    subsystem="inference",
                    detail="radiometric calibration produced NaN/Inf",
                )
            )
            continue
        cal_bands: np.ndarray = calib_result.value

        # Band selection: B2, B3, B4, B8
        selected = select_bands(cal_bands, list(config.inference.input_bands))

        # Quality flags
        quality_flags = compute_quality_flags(
            bands=selected,
            exposure_us=raw_msg.exposure_us,
            utc_timestamp=raw_msg.timestamp_utc,
            cfg=config.preprocessing,
        )

        # --- Inference ---
        processed_frame = ProcessedFrameMsg(
            msg_type=MessageType.PROCESSED_FRAME,
            timestamp_utc=raw_msg.timestamp_utc,
            frame_id=raw_msg.frame_id,
            tensor=selected,
            quality_flags=frozenset(quality_flags),
            crop_origin_px=(0, 0),
            scale_factor=1.0,
        )
        result = engine.run(processed_frame)
        if not isinstance(result, Ok):
            fault_queue.put(
                FaultEventMsg(
                    msg_type=MessageType.FAULT_EVENT,
                    timestamp_utc=utc_now_iso(),
                    fault_code=result.error,  # type: ignore[union-attr]
                    subsystem="inference",
                    detail=f"inference failed for frame_id={raw_msg.frame_id}",
                )
            )
            continue

        inference_msg: InferenceResultMsg = result.value  # type: ignore[union-attr]

        # Route to controller
        try:
            inference_queue.put_nowait(inference_msg)
        except Exception:
            pass  # controller queue full; drop frame

        # Route to storage
        usability = (
            FrameUsabilityTag.TRAINING
            if not quality_flags
            else FrameUsabilityTag.INVALID
        )
        storage_msg = StorageWriteMsg(
            msg_type=MessageType.STORAGE_WRITE,
            timestamp_utc=raw_msg.timestamp_utc,
            frame_id=raw_msg.frame_id,
            raw_frame=raw_bands,
            processed_tensor=selected,
            inference_result=inference_msg,
            usability=usability,
        )
        try:
            storage_queue.put_nowait(storage_msg)
        except Exception:
            pass  # storage queue full

        frame_seq += 1

    _log.info("inference_process_stop")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(config_path: str = "config/default.toml") -> None:
    """PACT main entry point.

    Loads config, creates queues, spawns processes, and runs the mode management loop.

    Config is loaded and validated before any process is spawned.  A bad config raises
    immediately (the .unwrap() call below is the intended crash point).
    """

    # -----------------------------------------------------------------------
    # 1. Load and validate config
    # -----------------------------------------------------------------------
    config_result = load_config(config_path)
    if not isinstance(config_result, Ok):
        log.critical("config_load_failed", error=config_result.error)  # type: ignore[union-attr]
        sys.exit(1)
    config: PactConfig = config_result.value  # type: ignore[union-attr]
    log.info("config_loaded", config_path=config_path)

    # -----------------------------------------------------------------------
    # 2. Create all inter-process queues
    # -----------------------------------------------------------------------

    # imaging --> inference (preprocessing embedded in inference process)
    raw_frame_queue: multiprocessing.Queue[RawFrameMsg] = multiprocessing.Queue(maxsize=8)

    # inference --> controller
    inference_queue: multiprocessing.Queue[InferenceResultMsg] = multiprocessing.Queue(maxsize=16)

    # inference --> storage
    storage_queue: multiprocessing.Queue[StorageWriteMsg] = multiprocessing.Queue(maxsize=16)

    # controller --> gimbal hardware (stub — hardware driver not implemented in Phase I)
    gimbal_queue: multiprocessing.Queue[GimbalCommandMsg] = multiprocessing.Queue(maxsize=32)

    # controller + storage + telemetry --> comms (shared downlink priority queue)
    downlink_queue: queue.Queue[DownlinkItemMsg] = queue.Queue(maxsize=256)

    # comms --> ops main (model upload chunks)
    uplink_queue: queue.Queue[UploadChunkMsg] = queue.Queue(maxsize=32)

    # controller --> telemetry
    telemetry_queue: queue.Queue[object] = queue.Queue(maxsize=128)

    # any subsystem --> fault process
    fault_queue: multiprocessing.Queue[FaultEventMsg] = multiprocessing.Queue(maxsize=64)

    # any subsystem --> fault process (watchdog heartbeats)
    heartbeat_queue: multiprocessing.Queue[HeartbeatMsg] = multiprocessing.Queue(maxsize=128)

    # fault process --> ops main (mode transitions)
    mode_queue: multiprocessing.Queue[ModeChangeMsg] = multiprocessing.Queue(maxsize=16)

    log.info("queues_created")

    # -----------------------------------------------------------------------
    # 3. Create stop events (one per subsystem)
    # -----------------------------------------------------------------------

    imaging_stop = threading.Event()
    inference_stop = multiprocessing.Event()
    controller_stop = multiprocessing.Event()
    storage_stop = threading.Event()
    telemetry_stop = threading.Event()
    comms_stop = threading.Event()
    fault_stop = multiprocessing.Event()

    # -----------------------------------------------------------------------
    # 4. Spawn subsystem processes and threads
    # -----------------------------------------------------------------------

    # imaging — threading.Thread (I/O-bound GigE Vision capture)
    imaging_thread = threading.Thread(
        target=run_imaging_process,
        args=(config.fault, raw_frame_queue, fault_queue, heartbeat_queue, imaging_stop),
        name="imaging",
        daemon=True,
    )

    # inference — multiprocessing.Process (CPU-bound GPU inference; REQ-AIML-COMP-002)
    inference_proc = multiprocessing.Process(
        target=_run_inference_process,
        args=(
            config,
            raw_frame_queue,
            inference_queue,
            storage_queue,
            fault_queue,
            heartbeat_queue,
            inference_stop,
        ),
        name="inference",
        daemon=True,
    )

    # controller — multiprocessing.Process (safety-critical; isolated from GIL)
    controller_proc = multiprocessing.Process(
        target=run_controller_process,
        args=(
            config.controller,
            config.fault,
            inference_queue,
            telemetry_queue,
            fault_queue,
            heartbeat_queue,
            controller_stop,
        ),
        name="controller",
        daemon=True,
    )

    # storage — threading.Thread (I/O-bound disk writes)
    storage_thread = threading.Thread(
        target=run_storage_process,
        args=(
            config.storage,
            storage_queue,
            downlink_queue,
            fault_queue,
            heartbeat_queue,
            "data/flight/manifest.jsonl",
            storage_stop,
        ),
        name="storage",
        daemon=True,
    )

    # telemetry — threading.Thread (I/O-bound serialisation)
    telemetry_thread = threading.Thread(
        target=run_telemetry_process,
        args=(
            config.comms.ccsds_apid,
            telemetry_queue,
            downlink_queue,
            heartbeat_queue,
            telemetry_stop,
        ),
        name="telemetry",
        daemon=True,
    )

    # comms — threading.Thread (asyncio I/O-bound TDRSS link management)
    comms_thread = threading.Thread(
        target=run_comms_process,
        args=(
            config.comms,
            config.fault,
            downlink_queue,
            uplink_queue,
            fault_queue,
            heartbeat_queue,
            comms_stop,
        ),
        name="comms",
        daemon=True,
    )

    # fault — multiprocessing.Process (must be immune to GIL starvation)
    fault_proc = multiprocessing.Process(
        target=run_fault_process,
        args=(config.fault, heartbeat_queue, fault_queue, mode_queue, fault_stop),
        name="fault",
        daemon=True,
    )

    # Ordered start: fault monitor first so it is ready before any subsystem can emit faults
    fault_proc.start()
    storage_thread.start()
    telemetry_thread.start()
    comms_thread.start()
    imaging_thread.start()
    inference_proc.start()
    controller_proc.start()

    mp_processes: list[multiprocessing.Process] = [inference_proc, controller_proc, fault_proc]
    threads: list[threading.Thread] = [
        imaging_thread, storage_thread, telemetry_thread, comms_thread
    ]

    log.info("all_subsystems_started", process_count=len(mp_processes), thread_count=len(threads))

    # -----------------------------------------------------------------------
    # 5. Signal handlers for graceful shutdown
    # -----------------------------------------------------------------------

    def _shutdown(signum: int, frame: object) -> None:
        log.info("shutdown_requested", signal=signum)
        # Signal all subsystems to stop
        imaging_stop.set()
        storage_stop.set()
        telemetry_stop.set()
        comms_stop.set()
        inference_stop.set()
        controller_stop.set()
        fault_stop.set()
        # Join threads with timeout
        for t in threads:
            t.join(timeout=5.0)
        # Terminate processes
        for p in mp_processes:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # -----------------------------------------------------------------------
    # 6. Mode management loop (ops main thread)
    # -----------------------------------------------------------------------

    current_mode: SystemMode = SystemMode.IDLE
    upload_session: Optional[ModelUploadSession] = None
    log.info("mode_initialized", mode=current_mode.value)

    while True:
        # --- Monitor process liveness ---
        for p in mp_processes:
            if not p.is_alive() and p.exitcode is not None:
                log.error("process_died", name=p.name, exitcode=p.exitcode)
                try:
                    fault_queue.put_nowait(
                        FaultEventMsg(
                            msg_type=MessageType.FAULT_EVENT,
                            timestamp_utc=utc_now_iso(),
                            fault_code=FaultCode.PROCESS_DIED,
                            subsystem=p.name,
                            detail=f"process '{p.name}' exited with code {p.exitcode}",
                        )
                    )
                except Exception:
                    pass

        # --- Drain mode_queue and apply validated transitions ---
        try:
            mode_msg: ModeChangeMsg = mode_queue.get_nowait()
            result = transition_mode(current_mode, mode_msg.new_mode)
            if isinstance(result, Ok):
                prev_mode = current_mode
                current_mode = result.value  # type: ignore[union-attr]
                log.info(
                    "mode_transition",
                    from_mode=prev_mode.value,
                    to_mode=current_mode.value,
                    requested_by=mode_msg.requested_by,
                )
                if current_mode == SystemMode.SAFE:
                    log.warning("safe_mode_entered", requested_by=mode_msg.requested_by)
            else:
                log.warning(
                    "invalid_mode_transition",
                    error=result.error,  # type: ignore[union-attr]
                    requested_by=mode_msg.requested_by,
                )
        except queue.Empty:
            pass

        # --- Route uplink chunks to model deployment logic ---
        try:
            while True:
                chunk: UploadChunkMsg = uplink_queue.get_nowait()
                if chunk.chunk_index == 0 or upload_session is None:
                    upload_session = ModelUploadSession(
                        total_chunks=chunk.total_chunks,
                        received_chunks=frozenset(),
                        expected_crc32=chunk.expected_crc32,
                        staged_path=config.comms.staged_model_path,
                        deploy_state=ModelDeployState.STAGED,
                    )
                    log.info("model_upload_session_started", total_chunks=chunk.total_chunks)

                chunk_result = process_uplink_chunk(upload_session, chunk)
                if isinstance(chunk_result, Ok):
                    upload_session = chunk_result.value  # type: ignore[union-attr]
                    if upload_session.deploy_state == ModelDeployState.ACTIVE:
                        # All chunks received and CRC verified — activate the model
                        act_result = activate_staged_model(
                            upload_session,
                            config.inference.model_path,
                            config.inference.rollback_model_path,
                        )
                        if isinstance(act_result, Ok):
                            log.info("model_activated_successfully")
                            upload_session = None
                        else:
                            log.error("model_activation_failed", error=act_result.error)  # type: ignore[union-attr]
                            upload_session = None
                else:
                    log.error("uplink_chunk_error", error=chunk_result.error)  # type: ignore[union-attr]
                    upload_session = None
        except queue.Empty:
            pass

        time.sleep(1.0)


if __name__ == "__main__":
    main()
