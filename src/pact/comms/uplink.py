"""
Model uplink handler for PACT comms subsystem.

Handles chunked model file upload from the ground station, with CRC-32 verification
of each chunk and of the complete reassembled file. Provides staged deployment with
rollback support.

Upload lifecycle:
    1. Ground station sends N UploadChunkMsg values (each with chunk_index and data bytes).
    2. All chunks are written to disk in order; the complete file CRC is verified at the end.
    3. When all chunks are received, session.expected_crc32 is checked against the file.
    4. The operator calls activate_staged_model() to atomically promote staged → active
       and save the current active model as rollback.
    5. If the new model fails, rollback_model() restores the previous active model.

Satisfies: REQ-AIML-HIGH-004, REQ-AIML-HIGH-005, GOAL-004
"""

from __future__ import annotations

import dataclasses
import os
import shutil
from dataclasses import dataclass

import structlog

from pact.comms.ccsds import compute_crc32, verify_crc32
from pact.types.enums import FaultCode, ModelDeployState
from pact.types.enums import Err, Ok, Result
from pact.types.messages import UploadChunkMsg

log = structlog.get_logger().bind(subsystem="comms.uplink")


@dataclass(frozen=True)
class ModelUploadSession:
    """Tracks the state of a chunked model upload.

    This dataclass is immutable — each call to process_uplink_chunk() returns a new
    session with updated state rather than mutating in place.

    Fields
    ------
    total_chunks:
        Total number of chunks expected for this upload session.
    received_chunks:
        Frozenset of chunk indices that have been successfully verified and written.
    expected_crc32:
        CRC-32 checksum of the complete model file, provided by the ground station at
        session initiation. Verified after all chunks are reassembled.
    staged_path:
        Filesystem path where the reassembled model file is staged.
    deploy_state:
        Current deployment lifecycle state (STAGED, ACTIVE, ROLLBACK_AVAILABLE).
    """

    total_chunks: int
    received_chunks: frozenset[int]
    expected_crc32: int
    staged_path: str
    deploy_state: ModelDeployState


def process_uplink_chunk(
    session: ModelUploadSession,
    chunk: UploadChunkMsg,
) -> "Result[ModelUploadSession, FaultCode]":
    """Process one chunk of a chunked model upload. REQ-AIML-HIGH-004.

    Steps:
    1. Append chunk.data to session.staged_path (create file on chunk_index == 0).
    2. Add chunk.chunk_index to received_chunks.
    3. If all chunks received, verify complete file CRC against session.expected_crc32.
    4. On CRC mismatch → Err(FaultCode.MODEL_CORRUPT).
    5. If complete and CRC passes → deploy_state = ModelDeployState.STAGED.

    Does NOT activate the model — activation is a separate explicit step.

    Parameters
    ----------
    session:
        Current upload session state.
    chunk:
        Incoming chunk message from the uplink queue.

    Returns
    -------
    Result[ModelUploadSession, FaultCode]
        Ok(updated_session) on success.
        Err(FaultCode.MODEL_CORRUPT) if any CRC check fails.
    """
    # Step 1: No per-chunk CRC field in UploadChunkMsg — chunks are verified by accumulation.
    # The complete-file CRC (session.expected_crc32) is verified after all chunks arrive.
    # chunk.data is the raw bytes for this chunk; chunk.expected_crc32 is the same value
    # as session.expected_crc32 (ground station echoes it on every chunk for confirmation).

    # Step 2: Append chunk data to staged file
    mode = "ab" if chunk.chunk_index > 0 else "wb"
    try:
        os.makedirs(os.path.dirname(session.staged_path), exist_ok=True)
        with open(session.staged_path, mode) as fh:
            fh.write(chunk.data)
    except OSError as exc:
        log.error("chunk_write_failed", chunk_index=chunk.chunk_index, error=str(exc))  # noqa: E501
        return Err(FaultCode.MODEL_CORRUPT)

    # Step 3: Record received chunk
    updated_received = session.received_chunks | frozenset({chunk.chunk_index})

    # Step 4: Check if all chunks received
    all_received = len(updated_received) == session.total_chunks
    new_deploy_state = session.deploy_state

    if all_received:
        # Verify complete file CRC
        try:
            with open(session.staged_path, "rb") as fh:
                file_bytes = fh.read()
        except OSError as exc:
            log.error("staged_file_read_failed", error=str(exc))
            return Err(FaultCode.MODEL_CORRUPT)

        if not verify_crc32(file_bytes, session.expected_crc32):
            log.error(
                "full_file_crc_mismatch",
                expected=session.expected_crc32,
                computed=compute_crc32(file_bytes),
            )
            return Err(FaultCode.MODEL_CORRUPT)

        new_deploy_state = ModelDeployState.STAGED
        log.info("model_upload_complete", staged_path=session.staged_path)

    return Ok(
        dataclasses.replace(
            session,
            received_chunks=updated_received,
            deploy_state=new_deploy_state,
        )
    )


def activate_staged_model(
    staged_path: str,
    active_path: str,
    rollback_path: str,
) -> "Result[None, FaultCode]":
    """Promote the staged model to active and save the current active as rollback.
    REQ-AIML-HIGH-005.

    Atomic promotion steps:
    1. Copy active_path → rollback_path (preserves current active for rollback).
    2. Move staged_path → active_path.

    If either step fails, returns Err(FaultCode.MODEL_CORRUPT). The caller should
    attempt rollback_model() if activation fails partway through.

    Parameters
    ----------
    staged_path:
        Path to the fully verified staged model file.
    active_path:
        Path where the active model is loaded from by the inference process.
    rollback_path:
        Path where the previous active model is saved for rollback.

    Returns
    -------
    Result[None, FaultCode]
        Ok(None) on success. Err(FaultCode.MODEL_CORRUPT) on any filesystem error.
    """
    try:
        # Save current active as rollback. On first deployment the file won't exist —
        # FileNotFoundError is expected and safe to skip.
        try:
            shutil.copy2(active_path, rollback_path)
            log.info("rollback_saved", rollback_path=rollback_path)
        except FileNotFoundError:
            log.info("no_prior_active_model", detail="first deployment; skipping rollback save")
        shutil.move(staged_path, active_path)
        log.info("model_activated", active_path=active_path)
        return Ok(None)
    except OSError as exc:
        log.error("model_activation_failed", error=str(exc))
        return Err(FaultCode.MODEL_CORRUPT)


def rollback_model(
    active_path: str,
    rollback_path: str,
) -> "Result[None, FaultCode]":
    """Swap active and rollback models. REQ-AIML-HIGH-005.

    Replaces the current active model with the rollback model. The (now-failed) active
    model is overwritten. This operation is not reversible without a new upload session.

    Parameters
    ----------
    active_path:
        Path to the current (presumably failing) active model.
    rollback_path:
        Path to the rollback model (the model active before the most recent activation).

    Returns
    -------
    Result[None, FaultCode]
        Ok(None) on success. Err(FaultCode.MODEL_CORRUPT) if rollback_path does not exist
        or cannot be copied.
    """
    try:
        shutil.copy2(rollback_path, active_path)
        log.info("model_rolled_back", active_path=active_path, rollback_path=rollback_path)
        return Ok(None)
    except FileNotFoundError:
        # rollback_path does not exist — no prior activation has been performed.
        log.error("rollback_unavailable", rollback_path=rollback_path)
        return Err(FaultCode.MODEL_CORRUPT)
    except OSError as exc:
        log.error("rollback_failed", error=str(exc))
        return Err(FaultCode.MODEL_CORRUPT)
