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
import multiprocessing
import queue
import signal
import sys
import time
from typing import Optional

# internal
from pact.types.enums import MessageType, SystemMode
from pact.types.messages import (
    DownlinkItemMsg,
    FaultEventMsg,
    GimbalCommandMsg,
    HeartbeatMsg,
    InferenceResultMsg,
    ModeChangeMsg,
    RawFrameMsg,
    StorageWriteMsg,
    UploadChunkMsg,
)
from pact.ops.config_loader import load_config
from pact.ops.modes import VALID_TRANSITIONS, transition_mode

import structlog

log = structlog.get_logger().bind(subsystem="ops")


def main(config_path: str = "config/default.toml") -> None:
    """PACT main entry point.

    Loads config, creates queues, spawns processes, and runs the mode management loop.

    Config is loaded and validated before any process is spawned.  A bad config raises
    immediately (the .unwrap() call below is the intended crash point).

    # TODO: implement graceful shutdown on SIGTERM/SIGINT (currently processes are
    #        terminated with .terminate() after a join timeout).
    # TODO: implement process restart on crash (Phase II).
    # TODO: route UploadChunkMsg from uplink_queue to model deployment logic.
    """

    # -----------------------------------------------------------------------
    # 1. Load and validate config
    # -----------------------------------------------------------------------
    config_result = load_config(config_path)
    if isinstance(config_result, type(config_result)) and hasattr(config_result, "error"):
        # Err case
        log.critical("config_load_failed", error=config_result.error)  # type: ignore[attr-defined]
        sys.exit(1)
    # Ok case
    config = config_result.value  # type: ignore[attr-defined]
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

    # controller --> gimbal hardware (stub -- hardware driver not implemented in Phase I)
    gimbal_queue: multiprocessing.Queue[GimbalCommandMsg] = multiprocessing.Queue(maxsize=32)

    # controller + storage + telemetry --> comms (shared downlink priority queue)
    # Using a plain multiprocessing.Queue; comms_process applies priority ordering internally.
    downlink_queue: multiprocessing.Queue[DownlinkItemMsg] = multiprocessing.Queue(maxsize=256)

    # comms --> ops main (model upload chunks)
    uplink_queue: multiprocessing.Queue[UploadChunkMsg] = multiprocessing.Queue(maxsize=32)

    # controller --> telemetry
    telemetry_queue: multiprocessing.Queue[  # type: ignore[type-arg]
        object
    ] = multiprocessing.Queue(maxsize=128)

    # any subsystem --> fault process
    fault_queue: multiprocessing.Queue[FaultEventMsg] = multiprocessing.Queue(maxsize=64)

    # any subsystem --> fault process (watchdog heartbeats)
    heartbeat_queue: multiprocessing.Queue[HeartbeatMsg] = multiprocessing.Queue(maxsize=128)

    # fault process --> ops main (mode transitions)
    mode_queue: multiprocessing.Queue[ModeChangeMsg] = multiprocessing.Queue(maxsize=16)

    log.info("queues_created")

    # -----------------------------------------------------------------------
    # 3. Spawn subsystem processes
    # -----------------------------------------------------------------------

    # TODO: spawn imaging process
    # from pact.imaging.process import run_imaging_process
    # imaging_proc = multiprocessing.Process(
    #     target=run_imaging_process,
    #     args=(config.inference, raw_frame_queue, fault_queue, heartbeat_queue),
    #     name="imaging",
    #     daemon=True,
    # )

    # TODO: spawn inference process (includes preprocessing on hot path)
    # from pact.model.inference import run_inference_process
    # inference_proc = multiprocessing.Process(
    #     target=run_inference_process,
    #     args=(config.inference, raw_frame_queue, inference_queue, storage_queue,
    #           fault_queue, heartbeat_queue),
    #     name="inference",
    #     daemon=True,
    # )

    # TODO: spawn controller process
    # from pact.controller.process import run_controller_process
    # controller_proc = multiprocessing.Process(
    #     target=run_controller_process,
    #     args=(config.controller, inference_queue, gimbal_queue, telemetry_queue,
    #           fault_queue, heartbeat_queue),
    #     name="controller",
    #     daemon=True,
    # )

    # TODO: spawn storage process (threading.Thread inside its own process)
    # from pact.storage.process import run_storage_process
    # storage_proc = multiprocessing.Process(
    #     target=run_storage_process,
    #     args=(config.storage, storage_queue, downlink_queue, fault_queue, heartbeat_queue,
    #           "data/flight/manifest.jsonl"),
    #     name="storage",
    #     daemon=True,
    # )

    # TODO: spawn comms process
    # from pact.comms.process import run_comms_process
    # comms_proc = multiprocessing.Process(
    #     target=run_comms_process,
    #     args=(config.comms, downlink_queue, uplink_queue, fault_queue, heartbeat_queue),
    #     name="comms",
    #     daemon=True,
    # )

    # TODO: spawn telemetry process (threading.Thread)
    # from pact.telemetry.reporter import run_telemetry_process
    # telemetry_proc = multiprocessing.Process(
    #     target=run_telemetry_process,
    #     args=(config.comms.ccsds_apid, telemetry_queue, downlink_queue, heartbeat_queue),
    #     name="telemetry",
    #     daemon=True,
    # )

    # TODO: spawn fault process
    # from pact.fault.process import run_fault_process
    # fault_proc = multiprocessing.Process(
    #     target=run_fault_process,
    #     args=(config.fault, heartbeat_queue, fault_queue, mode_queue),
    #     name="fault",
    #     daemon=True,
    # )

    # processes: list[multiprocessing.Process] = [
    #     imaging_proc,
    #     inference_proc,
    #     controller_proc,
    #     storage_proc,
    #     comms_proc,
    #     telemetry_proc,
    #     fault_proc,
    # ]
    # for p in processes:
    #     p.start()
    # log.info("all_processes_started", count=len(processes))

    # -----------------------------------------------------------------------
    # 4. Mode management loop
    # -----------------------------------------------------------------------

    current_mode: SystemMode = SystemMode.IDLE
    log.info("mode_initialized", mode=current_mode.value)

    # TODO: implement SIGTERM/SIGINT handler for graceful shutdown

    while True:
        # Drain mode_queue and apply validated transitions.
        try:
            mode_msg: ModeChangeMsg = mode_queue.get(timeout=1.0)
            result = transition_mode(current_mode, mode_msg.new_mode)
            if hasattr(result, "value"):
                # Ok case
                prev_mode = current_mode
                current_mode = result.value  # type: ignore[attr-defined]
                log.info(
                    "mode_transition",
                    from_mode=prev_mode.value,
                    to_mode=current_mode.value,
                    requested_by=mode_msg.requested_by,
                )
            else:
                # Err case
                log.warning(
                    "invalid_mode_transition",
                    error=result.error,  # type: ignore[attr-defined]
                    requested_by=mode_msg.requested_by,
                )
        except queue.Empty:
            pass

        # TODO: drain uplink_queue and route UploadChunkMsg to model deployment logic
        # TODO: monitor process liveness and emit PROCESS_DIED fault if a child exits
        # TODO: implement clean shutdown when mode transitions to SAFE


if __name__ == "__main__":
    main()
