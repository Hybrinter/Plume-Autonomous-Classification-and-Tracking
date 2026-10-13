"""Verified run, checkpoint and dataset input loading for model analysis.

The loader fails closed on unsafe or inconsistent sources before any model
construction: source/output nesting, existing outputs, linked source
components, running executions and missing recorded checkpoints are all
rejected. Every source file is read exactly once; the bytes that were
validated are the bytes snapshotted for the bundle, and history is parsed
from a private frozen copy so a mid-load mutation cannot swap validated
content. The selected checkpoint's actual bytes are hashed before
deserialization, and the same verified bytes are deserialized through
``torch.load(weights_only=True)`` on a ``BytesIO`` so no second disk read
can race the check. Before returning, every source file is re-read once
and must equal the validated bytes. Checkpoint bytes stay out of the
bundle.

Contains:
  - load_model_inputs: the ``Result`` input boundary for ``analyze_model``.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import tomllib
from dataclasses import asdict, replace
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from flight.libs.types import Err, Ok, Result
from pydantic import TypeAdapter

from tools.ml_models.analysis.artifacts import BundleFile, dataset_identity
from tools.ml_models.analysis.config import ModelAnalysisConfig
from tools.ml_models.analysis.contracts import SplitEvidence, check_bundle_path
from tools.ml_models.analysis.model_measurement import ModelInputs
from tools.ml_models.analysis.training import (
    CheckpointRecord,
    EvaluationRecord,
    HistoryRecord,
    TrainingExecution,
    TrainingHistory,
    read_training_history,
)
from tools.ml_models.dataset.manifest import (
    SUPPORTED_SCHEMA_VERSIONS,
    DatasetManifest,
)
from tools.ml_models.train.config import TrainConfig
from tools.ml_models.train.provenance import training_provenance

if TYPE_CHECKING:
    from tools.ml_models.analysis.contracts import DatasetIdentity

_CHECKPOINT_FIELDS = frozenset(
    {
        "kind",
        "arch",
        "state_dict",
        "epoch",
        "conditioning",
        "config",
        "provenance",
        "dataset_hash",
        "step",
        "samples_seen",
        "metric",
        "direction",
        "value",
        "validation",
    }
)


def _contained(path: Path, root: Path) -> bool:
    """True when resolved ``path`` lies at or below resolved ``root``."""
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError, ValueError, RuntimeError:
        return True


def _linked(path: Path) -> bool:
    """True when any component of ``path``, including ancestors, is a link."""
    current = path
    while True:
        if os.path.islink(current) or os.path.isjunction(current):
            return True
        if current.parent == current:
            return False
        current = current.parent


def _linked_tree(root: Path) -> bool:
    """True when ``root`` or any descendant entry is a symlink or junction."""
    if _linked(root):
        return True
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            return True
        for entry in entries:
            if os.path.islink(entry) or os.path.isjunction(entry):
                return True
            try:
                if entry.is_dir():
                    stack.append(entry)
            except OSError:
                return True
    return False


def _canonical(value: object) -> str:
    """Canonical JSON; normalizes tuple/list and key order for recorded payloads."""
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _read_source(path: Path) -> Result[bytes, str]:
    """Read one source file exactly once, rejecting linked or missing files."""
    if _linked(path):
        return Err(f"source {path} contains a symlink or junction component")
    try:
        if not path.is_file():
            return Err(f"source {path} is missing")
        return Ok(path.read_bytes())
    except OSError as exc:
        return Err(f"cannot read source {path}: {exc}")


def _parse_train_config(raw: bytes) -> Result[TrainConfig, str]:
    """Parse the run's flat ``config.toml`` bytes strictly as ``TrainConfig``."""
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        return Err(f"invalid TOML in run config.toml: {exc}")
    try:
        return Ok(TypeAdapter(TrainConfig).validate_python(data))
    except ValueError as exc:
        return Err(f"invalid training config: {exc}")


def _parse_manifest(raw: bytes, path: Path) -> Result[DatasetManifest, str]:
    """Validate already-read manifest bytes; content hash is verified separately."""
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return Err(f"invalid dataset manifest {path}: {exc}")
    if not isinstance(payload, dict):
        return Err(f"{path.name} must be a JSON object")
    if "schema" not in payload:
        return Err(f"{path.name} missing schema")
    payload = dict(payload)
    payload["schema_version"] = payload.pop("schema")
    if payload["schema_version"] not in SUPPORTED_SCHEMA_VERSIONS:
        return Err(f"dataset schema {payload['schema_version']} is unsupported")
    try:
        return Ok(TypeAdapter(DatasetManifest).validate_python(payload))
    except ValueError as exc:
        return Err(f"invalid dataset manifest {path}: {exc}")


def _verified_dataset(
    root: Path,
) -> Result[tuple[DatasetIdentity, DatasetManifest, bytes], str]:
    """Verify content and manifest hash; return identity, manifest and manifest bytes."""
    if _linked_tree(root):
        return Err(f"dataset {root} contains a symlink or junction component")
    manifest_path = root / "dataset.json"
    if _linked(manifest_path):
        return Err(f"dataset manifest {manifest_path} contains a link component")
    identity = dataset_identity(root)
    if isinstance(identity, Err):
        return identity
    manifest_bytes = _read_source(manifest_path)
    if isinstance(manifest_bytes, Err):
        return manifest_bytes
    if hashlib.sha256(manifest_bytes.value).hexdigest() != identity.value.manifest_hash:
        return Err(f"dataset manifest {manifest_path} changed during verification")
    manifest = _parse_manifest(manifest_bytes.value, manifest_path)
    if isinstance(manifest, Err):
        return manifest
    if manifest.value.dataset_hash != identity.value.content_hash:
        return Err(f"dataset {root} content hash disagrees with its verified identity")
    return Ok((identity.value, manifest.value, manifest_bytes.value))


def _verify_checkpoint_payload(
    payload: object,
    record: CheckpointRecord,
    records: tuple[HistoryRecord, ...],
    execution: TrainingExecution,
) -> Result[dict[str, object], str]:
    """Require serialized fields to agree with the recorded identity and execution."""
    if not isinstance(payload, dict) or not _CHECKPOINT_FIELDS.issubset(payload):
        return Err("checkpoint payload is missing required serialized fields")
    for name in ("epoch", "step", "samples_seen"):
        counter = payload[name]
        if not isinstance(counter, int) or isinstance(counter, bool) or counter < 0:
            return Err(f"checkpoint payload {name} must be a real nonnegative integer")
    identity = record.identity
    expectations = (
        (payload["kind"], identity.kind, "kind"),
        (payload["arch"], identity.arch, "arch"),
        (payload["epoch"], identity.epoch, "epoch"),
        (payload["step"], identity.step, "step"),
        (payload["dataset_hash"], identity.training_dataset_hash, "dataset_hash"),
        (payload["conditioning"], execution.conditioning, "conditioning"),
        (payload["metric"], execution.metric, "metric"),
    )
    if any(actual != expected for actual, expected, _ in expectations):
        names = [name for actual, expected, name in expectations if actual != expected]
        return Err(f"checkpoint payload disagrees with recorded identity: {sorted(names)}")
    if _canonical(payload["config"]) != _canonical(execution.config) or _canonical(
        payload["provenance"]
    ) != _canonical(execution.provenance):
        return Err("checkpoint config or provenance disagrees with the recorded execution")
    matching = tuple(
        entry
        for entry in records
        if isinstance(entry, EvaluationRecord) and entry.checkpoint.identity == identity
    )
    if matching and payload["samples_seen"] != matching[-1].samples_seen:
        return Err("checkpoint samples_seen disagrees with its durable evaluation record")
    try:
        validation = TypeAdapter(SplitEvidence).validate_python(payload["validation"])
    except (ValueError, TypeError) as exc:
        return Err(f"checkpoint validation evidence is malformed: {exc}")
    if validation != replace(record.validation, checkpoint_hash=None):
        return Err("checkpoint validation evidence disagrees with the recorded checkpoint")
    return Ok(payload)


def _frozen_history(execution: bytes, history: bytes | None) -> Result[TrainingHistory, str]:
    """Parse the run history from a private copy of the already-read bytes."""
    try:
        with tempfile.TemporaryDirectory(prefix=".model-history-") as staging:
            staged = Path(staging)
            (staged / "execution.json").write_bytes(execution)
            if history is not None:
                (staged / "history.jsonl").write_bytes(history)
            return read_training_history(staged)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        return Err(f"cannot stage frozen run history: {exc}")


def load_model_inputs(cfg: ModelAnalysisConfig) -> Result[ModelInputs, str]:
    """Verify and load every frozen input ``measure_model`` consumes.

    ``cfg.checkpoint`` selects exactly one recorded best/last record; its
    file bytes are SHA-verified before deserialization. Datasets are
    verified through ``load_manifest`` and bound to the recorded manifest
    hash. An external ``cfg.dataset`` override must share the training
    bands, normalization unit and GSD reference. Returns the typed
    ``ModelInputs`` or an explicit error without creating output.
    """
    import torch

    from tools.ml_models.arch.film import (
        CONDITIONING_ID,
        IGNORED_CONDITIONING_ID,
        IgnoreGsd,
    )
    from tools.ml_models.arch.registry import build as build_model
    from tools.ml_models.train.losses import build_loss

    try:
        run = Path(cfg.run)
        out = Path(cfg.out)
        if cfg.checkpoint not in ("best", "last"):
            return Err("analysis requires an explicit best or last checkpoint selector")
        if os.path.lexists(out):
            return Err(f"analysis output {out} already exists; refusing to overwrite")
        if not run.is_dir():
            return Err(f"run directory {run} does not exist")
        if _linked(run):
            return Err(f"run path {run} contains a symlink or junction component")

        config_path = run / "config.toml"
        execution_path = run / "execution.json"
        history_path = run / "history.jsonl"
        config_bytes = _read_source(config_path)
        if isinstance(config_bytes, Err):
            return config_bytes
        execution_bytes = _read_source(execution_path)
        if isinstance(execution_bytes, Err):
            return execution_bytes
        history_bytes: bytes | None = None
        if history_path.exists() or os.path.islink(history_path):
            read = _read_source(history_path)
            if isinstance(read, Err):
                return read
            history_bytes = read.value

        history = _frozen_history(execution_bytes.value, history_bytes)
        if isinstance(history, Err):
            return history
        execution = history.value.execution
        if execution.status == "RUNNING":
            return Err(f"run {run} is still executing; wait for a terminal status")
        record = execution.best if cfg.checkpoint == "best" else execution.last
        if record is None:
            return Err(f"run {run} has no recorded {cfg.checkpoint} checkpoint")
        if record.path != "checkpoints/" + cfg.checkpoint + ".pt":
            return Err(f"recorded {cfg.checkpoint} checkpoint path is inconsistent")

        train_config = _parse_train_config(config_bytes.value)
        if isinstance(train_config, Err):
            return train_config
        if _canonical(asdict(train_config.value)) != _canonical(execution.config):
            return Err("run config.toml disagrees with the recorded execution config")

        train_root = Path(train_config.value.dataset)
        recorded_dataset = execution.provenance.get("dataset")
        recorded_path = recorded_dataset.get("path") if isinstance(recorded_dataset, dict) else None
        if not isinstance(recorded_path, str):
            return Err("recorded provenance lacks a training dataset path")
        if str(Path(recorded_path)) != str(train_root):
            return Err("recorded training dataset path disagrees with run config.toml")
        eval_root = Path(cfg.dataset) if cfg.dataset is not None else train_root
        if _contained(out, run) or _contained(out, train_root) or _contained(out, eval_root):
            return Err(f"analysis output {out} lies inside a source run or dataset")
        training = _verified_dataset(train_root)
        if isinstance(training, Err):
            return training
        train_identity, train_manifest, train_manifest_bytes = training.value
        if train_identity.content_hash != execution.dataset_hash:
            return Err("training dataset content differs from the recorded execution")
        manifest_hash = record.validation.dataset_manifest_hash
        if manifest_hash is None:
            return Err(
                "the recorded checkpoint predates dataset manifest hashing; "
                "retrain or rebuild the run so validation evidence carries a manifest hash"
            )
        if train_identity.manifest_hash != manifest_hash:
            return Err("training dataset manifest differs from the recorded validation hash")
        evaluation = _verified_dataset(eval_root)
        if isinstance(evaluation, Err):
            return evaluation
        eval_identity, eval_manifest, eval_manifest_bytes = evaluation.value
        if cfg.dataset is not None and (
            eval_manifest.band_names != train_manifest.band_names
            or eval_manifest.norm != train_manifest.norm
            or eval_manifest.image_dtype != train_manifest.image_dtype
            or eval_manifest.gsd_reference_m != train_manifest.gsd_reference_m
        ):
            return Err(
                "external evaluation dataset must keep the training bands, "
                "order, unit normalization and GSD reference"
            )
        try:
            provenance = training_provenance(train_root, train_manifest, execution.kind)
        except (OSError, ValueError, RuntimeError) as exc:
            return Err(f"cannot recompute training provenance: {exc}")
        if _canonical(provenance) != _canonical(execution.provenance):
            return Err("recomputed training provenance disagrees with the recorded execution")

        checkpoint_path = run.joinpath(*record.path.split("/"))
        checkpoint_bytes = _read_source(checkpoint_path)
        if isinstance(checkpoint_bytes, Err):
            return checkpoint_bytes
        if hashlib.sha256(checkpoint_bytes.value).hexdigest() != record.identity.sha256:
            return Err(
                f"checkpoint {checkpoint_path} bytes differ from the recorded identity; "
                "refusing to analyze a different checkpoint than the one recorded"
            )
        try:
            payload: object = torch.load(
                BytesIO(checkpoint_bytes.value), map_location="cpu", weights_only=True
            )
        except Exception as exc:  # noqa: BLE001 - torch/pickle errors surface as Err
            return Err(f"cannot deserialize verified checkpoint bytes: {exc}")
        checked = _verify_checkpoint_payload(payload, record, history.value.records, execution)
        if isinstance(checked, Err):
            return checked

        try:
            model = build_model(execution.kind, execution.arch, len(train_manifest.band_names))
        except (ValueError, RuntimeError) as exc:
            return Err(f"cannot build recorded architecture {execution.arch!r}: {exc}")
        expected_conditioning = (
            IGNORED_CONDITIONING_ID if isinstance(model, IgnoreGsd) else CONDITIONING_ID
        )
        if expected_conditioning != execution.conditioning:
            return Err("built model conditioning disagrees with the recorded execution")
        state = checked.value.get("state_dict")
        if not isinstance(state, dict):
            return Err("checkpoint payload lacks a state_dict mapping")
        try:
            model.load_state_dict(state, strict=True)
            model.to(cfg.device)
        except (RuntimeError, ValueError, TypeError) as exc:
            return Err(f"checkpoint state does not load strictly: {exc}")
        model.eval()
        try:
            objective = build_loss(
                train_config.value.loss,
                pos_weight=train_config.value.pos_weight,
                focal_gamma=train_config.value.focal_gamma,
                focal_alpha=train_config.value.focal_alpha,
            )
        except ValueError as exc:
            return Err(f"cannot rebuild the recorded objective: {exc}")

        snapshot_sources = (
            (config_path, "source/config.toml", config_bytes.value),
            (execution_path, "source/execution.json", execution_bytes.value),
            (history_path, "source/history.jsonl", history_bytes),
            (train_root / "dataset.json", "source/training-dataset.json", train_manifest_bytes),
            (eval_root / "dataset.json", "source/evaluation-dataset.json", eval_manifest_bytes),
        )
        snapshots: list[BundleFile] = []
        for source_path, bundle_path, trusted in snapshot_sources:
            if trusted is None:
                continue
            try:
                check_bundle_path(bundle_path)
            except (ValueError, TypeError) as exc:
                return Err(f"unsafe snapshot path {bundle_path!r}: {exc}")
            snapshots.append(BundleFile(bundle_path, trusted))

        for source_path, _, trusted in snapshot_sources:
            if trusted is None:
                continue
            try:
                current = source_path.read_bytes()
            except OSError as exc:
                return Err(f"cannot re-verify source {source_path}: {exc}")
            if current != trusted:
                return Err(f"source {source_path} changed during input verification")
        try:
            if checkpoint_path.read_bytes() != checkpoint_bytes.value:
                return Err(f"checkpoint {checkpoint_path} changed during input verification")
        except OSError as exc:
            return Err(f"cannot re-verify checkpoint {checkpoint_path}: {exc}")

        return Ok(
            ModelInputs(
                history.value,
                record.identity,
                train_identity,
                eval_identity,
                train_root,
                eval_root,
                train_manifest,
                eval_manifest,
                model,
                objective,
                tuple(snapshots),
            )
        )
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError) as exc:
        return Err(f"model input loading failed before measurement: {exc}")
