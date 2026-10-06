"""Bounded prediction/evidence persistence boundary.

Evaluation hands per-row predictions plus evaluator-derived scalar records to
a ``CaptureSink`` implementation, which owns how much is persisted. Capture is
mechanical: it never derives scores, labels, groups, or bins, and it never
transforms the supplied arrays.

Contains:
  - CaptureRow: evaluator-supplied scalar record with selection flags and
    optional frozen spatial evidence.
  - CaptureSink: sink protocol with add/abort/close/references.
  - BoundedCaptureSink: disk sink with bounded preview selection and a byte
    budget covering persisted payload and cached arrays.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import bisect
import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import asdict, field
from pathlib import Path
from typing import BinaryIO, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from flight.libs.types import Err, Ok, Result
from pydantic import ConfigDict, StrictBool, model_validator
from pydantic.dataclasses import dataclass

from tools.ml_models.analysis.artifacts import INCOMPLETE_FILENAME, artifact_file_ref
from tools.ml_models.analysis.config import CaptureConfig
from tools.ml_models.analysis.contracts import (
    ArtifactKind,
    ArtifactRef,
    FiniteNumber,
    MetricValue,
    SampleKey,
)
from tools.ml_models.analysis.metrics.spatial import SpatialRow

_SCHEMA = ConfigDict(extra="forbid")

ROWS_FILENAME = "rows.jsonl"
METADATA_FILENAME = "metadata.json"
PREVIEWS_DIR = "previews"
FULL_DIR = "full"

type CaptureFamily = Literal["REPRESENTATIVE", "WORST", "FALSE_POSITIVE", "FALSE_NEGATIVE"]
_FAMILIES: tuple[CaptureFamily, ...] = (
    "REPRESENTATIVE",
    "WORST",
    "FALSE_POSITIVE",
    "FALSE_NEGATIVE",
)


@dataclass(frozen=True, slots=True, config=_SCHEMA)
class CaptureRow:
    """Evaluator-supplied scalar record for one scored sample.

    Attributes:
        key: Alignment key for the scored sample.
        group_id: Evaluator-assigned group identifier, stored unchanged.
        bin_id: Evaluator-assigned bin identifier, stored unchanged.
        label: Binary 0/1 sample label supplied by the evaluator.
        gsd_m: Strictly positive finite ground-sample distances in metres.
        metrics: Evaluator-derived metric values, stored unchanged.
        failure_score: Finite evaluator ranking score for worst-first
            selection.
        false_positive: Whether the row qualifies for the false-positive
            family.
        false_negative: Whether the row qualifies for the false-negative
            family.
        array_view: Array provenance tag; ``CANONICAL`` marks arrays in the
            canonical orientation regardless of the stored transform element.
        spatial: Frozen component-matching and boundary evidence for segmentor
            rows; ``None`` for classifier rows. Compact records only - no
            dense arrays are retained here.
    """

    key: SampleKey
    group_id: str
    bin_id: str
    label: FiniteNumber
    gsd_m: tuple[FiniteNumber, FiniteNumber]
    metrics: tuple[MetricValue, ...]
    failure_score: FiniteNumber
    false_positive: StrictBool = False
    false_negative: StrictBool = False
    array_view: Literal["CANONICAL"] = "CANONICAL"
    spatial: SpatialRow | None = field(default=None)

    @model_validator(mode="after")
    def _bounds(self) -> CaptureRow:
        if self.label not in (0.0, 1.0):
            raise ValueError("label must be exactly 0 or 1")
        if self.gsd_m[0] <= 0 or self.gsd_m[1] <= 0:
            raise ValueError("gsd_m components must be positive")
        names = [metric.name for metric in self.metrics]
        if len(set(names)) != len(names):
            raise ValueError("metric names must be unique")
        return self


@runtime_checkable
class CaptureSink(Protocol):
    """Destination for captured evidence rows."""

    def add(
        self,
        row: CaptureRow,
        *,
        image: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
        logits: npt.NDArray[np.float32],
    ) -> Result[None, str]:
        """Record one evaluated row and its dense arrays.

        Returns:
            Result[None, str]: Ok when the row is accepted, Err with the
            failure reason otherwise. Any failure makes the sink unusable.
        """
        ...

    def abort(self, reason: str) -> Result[None, str]:
        """Mark the capture failed without removing the incomplete marker.

        Returns:
            Result[None, str]: Ok after recording ``reason``; Err when the
            sink is already finished or failed.
        """
        ...

    def close(self) -> Result[None, str]:
        """Flush and finish the sink.

        Returns:
            Result[None, str]: Ok on a clean finish, Err with the failure
            reason otherwise.
        """
        ...

    def references(self) -> tuple[ArtifactRef, ...]:
        """Return artifact references; empty until a successful close."""
        ...


def _canonical_json(data: object) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _key_id(key: SampleKey) -> str:
    return hashlib.sha256(_canonical_json(asdict(key))).hexdigest()


def _rank_digest(seed: int, key: SampleKey) -> str:
    payload = {"seed": seed, "key": asdict(key)}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _npz_bytes(
    image: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
    logits: npt.NDArray[np.float32],
) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, image=image, target=target, logits=logits)
    return buffer.getvalue()


def _contained(path: Path, root: Path) -> bool:
    resolved_root = root.resolve()
    resolved = path.resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


def _check_arrays(
    key: SampleKey,
    image: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
    logits: npt.NDArray[np.float32],
) -> str | None:
    for name, array in (("image", image), ("target", target), ("logits", logits)):
        if array.dtype != np.dtype(np.float32):
            return f"{name} must have dtype float32"
        if any(dim <= 0 for dim in array.shape):
            return f"{name} has a nonpositive dimension; got {array.shape}"
        if not bool(np.isfinite(array).all()):
            return f"{name} contains nonfinite values"
    if image.ndim != 3:
        return f"image must have shape (C, H, W); got {image.shape}"
    _, height, width = image.shape
    if (height, width) != tuple(key.spatial_shard):
        return (
            f"image dimensions {(height, width)} do not match "
            f"key.spatial_shard {tuple(key.spatial_shard)}"
        )
    if not bool(((image >= 0) & (image <= 1)).all()):
        return "image values must lie in the unit interval [0, 1]"
    if not bool(((target == 0) | (target == 1)).all()):
        return "target values must be binary"
    if key.task == "classifier":
        if target.shape != (1,) or logits.shape != (1,):
            return (
                "classifier target and logits must be shape (1,); "
                f"got {target.shape} and {logits.shape}"
            )
        return None
    if target.shape != (1, height, width) or logits.shape != (1, height, width):
        return (
            f"segmentor target and logits must be shape (1, {height}, {width}); "
            f"got {target.shape} and {logits.shape}"
        )
    return None


def _selection_entry(
    key: str, families: list[str], file: str | None, reason: str | None
) -> dict[str, object]:
    return {
        "key": key,
        "families": families,
        "file": file,
        "status": "OMITTED" if reason is not None else "KEPT",
        "reason": reason,
    }


def _metadata_bytes(
    cfg: CaptureConfig,
    seen_rows: int,
    families: Mapping[str, list[str]],
    selected: list[dict[str, object]],
    omitted: int,
) -> bytes:
    metadata = {
        "array_view": "CANONICAL",
        "settings": {
            "retention": cfg.retention,
            "max_preview_images": cfg.max_preview_images,
            "max_capture_bytes": cfg.max_capture_bytes,
            "examples_per_family": cfg.examples_per_family,
            "seed": cfg.seed,
        },
        "seen_rows": seen_rows,
        "families": families,
        "selected": selected,
        "omitted_previews": omitted,
    }
    return _canonical_json(metadata)


def _metadata_reserve(cfg: CaptureConfig) -> int:
    dummy = "0" * 64
    entry = _selection_entry(dummy, list(_FAMILIES), f"{PREVIEWS_DIR}/{dummy}.npz", "byte_budget")
    families: dict[str, list[str]] = {name: [dummy] * cfg.examples_per_family for name in _FAMILIES}
    return len(
        _metadata_bytes(
            cfg,
            10**19,
            families,
            [entry] * cfg.max_preview_images,
            cfg.max_preview_images,
        )
    )


class BoundedCaptureSink:
    """Disk capture sink with bounded candidate memory and a byte budget.

    The scalar stream ``rows.jsonl`` and the final ``metadata.json`` are
    mandatory; exhausting the byte budget on them fails the sink and keeps
    the ``.incomplete`` marker. Preview payloads are best-effort: selected
    candidates that cannot be retained or written within the budget are
    recorded as omitted in the metadata, never silently dropped. ``FULL``
    retention additionally writes every row's dense arrays under ``full/``;
    those writes are mandatory and fail on budget exhaustion.
    """

    def __init__(self, out: Path, cfg: CaptureConfig, *, dataset: Path) -> None:
        self._out = Path(out)
        self._cfg = cfg
        source = Path(dataset).resolve()
        resolved_out = self._out.resolve()
        if _contained(resolved_out, source) or _contained(source, resolved_out):
            raise ValueError(f"capture output {out} overlaps the source dataset {dataset}")
        try:
            self._out.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise ValueError(f"capture output {out} already exists") from None
        self._marker = self._out / INCOMPLETE_FILENAME
        self._marker.touch(exist_ok=False)
        self._rows_handle: BinaryIO = (self._out / ROWS_FILENAME).open("xb")
        self._reserve = _metadata_reserve(cfg)
        self._persisted = 0
        self._cache: dict[str, tuple[npt.NDArray[np.float32], ...]] = {}
        self._cache_bytes = 0
        self._seen: set[str] = set()
        self._families: dict[str, list[tuple[float, str, str]]] = {name: [] for name in _FAMILIES}
        self._full_written: list[str] = []
        self._failed: str | None = None
        self._closed = False
        self._refs: tuple[ArtifactRef, ...] = ()

    @classmethod
    def create(
        cls, out: Path, cfg: CaptureConfig, *, dataset: Path
    ) -> Result[BoundedCaptureSink, str]:
        """Create a sink, reporting reservation/validation failures as Err."""
        try:
            return Ok(cls(out, cfg, dataset=dataset))
        except (OSError, ValueError) as exc:
            return Err(f"cannot create capture output: {exc}")

    def _check_state(self) -> str | None:
        if self._closed:
            return "capture sink is already closed"
        return self._failed

    def _fits(self, extra: int) -> bool:
        return (
            self._persisted + self._cache_bytes + extra + self._reserve
            <= self._cfg.max_capture_bytes
        )

    def _drop_cached(self, key_id: str) -> None:
        arrays = self._cache.pop(key_id, None)
        if arrays is not None:
            self._cache_bytes -= sum(int(array.nbytes) for array in arrays)

    def _eviction_order(self) -> list[str]:
        selected = self._select()
        selected_set = set(selected)
        order = sorted(key for key in self._cache if key not in selected_set)
        order.extend(reversed(selected))
        return order

    def _ensure_fits(self, extra: int) -> bool:
        for key_id in self._eviction_order():
            if self._fits(extra):
                return True
            self._drop_cached(key_id)
        return self._fits(extra)

    def _release_cache(self) -> None:
        self._cache.clear()
        self._cache_bytes = 0

    def add(
        self,
        row: CaptureRow,
        *,
        image: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
        logits: npt.NDArray[np.float32],
    ) -> Result[None, str]:
        """Persist the scalar row, update selection candidates, and (FULL)
        write dense arrays."""
        problem = self._check_state()
        if problem is not None:
            return Err(problem)
        result = self._add(row, image=image, target=target, logits=logits)
        if isinstance(result, Err):
            self._failed = result.error
        return result

    def _add(
        self,
        row: CaptureRow,
        *,
        image: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
        logits: npt.NDArray[np.float32],
    ) -> Result[None, str]:
        problem = _check_arrays(row.key, image, target, logits)
        if problem is not None:
            return Err(problem)
        key_id = _key_id(row.key)
        if key_id in self._seen:
            return Err(f"duplicate sample key {row.key}")
        try:
            line = _canonical_json(asdict(row)) + b"\n"
        except (TypeError, ValueError) as exc:
            return Err(f"capture row is not serializable: {exc}")
        if not self._ensure_fits(len(line)):
            return Err("capture byte budget exhausted writing scalar row")
        try:
            self._rows_handle.write(line)
            self._rows_handle.flush()
        except OSError as exc:
            return Err(f"scalar row write failed: {exc}")
        self._persisted += len(line)
        self._seen.add(key_id)
        if self._cfg.retention == "FULL":
            try:
                stored = _npz_bytes(image, target, logits)
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                return Err(f"dense prediction serialization failed: {exc}")
            if not self._ensure_fits(len(stored)):
                return Err("capture byte budget exhausted writing dense arrays")
            path = self._out / FULL_DIR / f"{key_id}.npz"
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as handle:
                    handle.write(stored)
            except OSError as exc:
                return Err(f"dense prediction write failed: {exc}")
            self._persisted += len(stored)
            self._full_written.append(f"{FULL_DIR}/{key_id}.npz")
        self._offer(row, key_id, image, target, logits)
        return Ok(None)

    def _offer(
        self,
        row: CaptureRow,
        key_id: str,
        image: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
        logits: npt.NDArray[np.float32],
    ) -> None:
        digest = _rank_digest(self._cfg.seed, row.key)
        families = self._families
        bisect.insort(families["REPRESENTATIVE"], (0.0, digest, key_id))
        ranked = (-row.failure_score, digest, key_id)
        bisect.insort(families["WORST"], ranked)
        if row.false_positive:
            bisect.insort(families["FALSE_POSITIVE"], ranked)
        if row.false_negative:
            bisect.insort(families["FALSE_NEGATIVE"], ranked)
        for members in families.values():
            del members[self._cfg.examples_per_family :]
        needed = {entry[-1] for members in families.values() for entry in members}
        for cached in list(self._cache):
            if cached not in needed:
                self._drop_cached(cached)
        if key_id in needed and key_id not in self._cache:
            nbytes = int(image.nbytes) + int(target.nbytes) + int(logits.nbytes)
            if self._fits(nbytes):
                self._cache[key_id] = (image.copy(), target.copy(), logits.copy())
                self._cache_bytes += nbytes

    def abort(self, reason: str) -> Result[None, str]:
        """Record a sticky failure; the incomplete marker is retained."""
        if self._closed:
            return Err("capture sink is already closed")
        if self._failed is None:
            self._failed = f"capture aborted: {reason}"
            self._release_cache()
            return Ok(None)
        return Err(self._failed)

    def close(self) -> Result[None, str]:
        """Finish selection, persist previews and metadata, release resources."""
        if self._closed:
            return Ok(None)
        try:
            self._rows_handle.close()
        except OSError as exc:
            if self._failed is None:
                self._failed = f"capture rows stream close failed: {exc}"
        if self._failed is not None:
            self._release_cache()
            return Err(self._failed)
        result = self._finish()
        if isinstance(result, Err):
            self._failed = result.error
            self._release_cache()
        return result

    def _finish(self) -> Result[None, str]:
        selected = self._select()
        selected_set = set(selected)
        for cached in list(self._cache):
            if cached not in selected_set:
                self._drop_cached(cached)
        memberships = self._memberships()
        payloads: dict[str, bytes] = {}
        kept: list[str] = []
        omitted: list[str] = []
        for key_id in selected:
            arrays = self._cache.get(key_id)
            if arrays is None:
                omitted.append(key_id)
            else:
                try:
                    payloads[key_id] = _npz_bytes(*arrays)
                except (OSError, ValueError, TypeError, RuntimeError) as exc:
                    return Err(f"preview serialization failed: {exc}")
                kept.append(key_id)
        while True:
            entries = [
                _selection_entry(
                    key_id,
                    memberships[key_id],
                    f"{PREVIEWS_DIR}/{key_id}.npz" if key_id in kept else None,
                    None if key_id in kept else "byte_budget",
                )
                for key_id in selected
            ]
            family_lists: dict[str, list[str]] = {
                name: [entry[-1] for entry in self._families[name]] for name in _FAMILIES
            }
            metadata = _metadata_bytes(
                self._cfg,
                len(self._seen),
                family_lists,
                entries,
                len(omitted),
            )
            total = (
                self._persisted
                + self._cache_bytes
                + sum(len(payloads[key_id]) for key_id in kept)
                + len(metadata)
            )
            if total <= self._cfg.max_capture_bytes:
                break
            if kept:
                victim = kept.pop()
                omitted.append(victim)
                del payloads[victim]
                self._drop_cached(victim)
            elif self._cache:
                self._drop_cached(next(iter(self._cache)))
            else:
                return Err("capture byte budget exhausted writing required metadata")
        previews_dir = self._out / PREVIEWS_DIR
        try:
            if kept:
                previews_dir.mkdir(parents=True, exist_ok=True)
                for key_id in kept:
                    with (previews_dir / f"{key_id}.npz").open("xb") as handle:
                        handle.write(payloads[key_id])
            with (self._out / METADATA_FILENAME).open("xb") as handle:
                handle.write(metadata)
        except OSError as exc:
            return Err(f"capture write failed; marker retained at {self._out}: {exc}")
        refs = self._build_refs(kept)
        if isinstance(refs, Err):
            return Err(refs.error)
        try:
            self._marker.unlink()
        except OSError as exc:
            return Err(f"capture marker removal failed: {exc}")
        self._refs = refs.value
        self._closed = True
        self._release_cache()
        return Ok(None)

    def _select(self) -> list[str]:
        selected: list[str] = []
        seen: set[str] = set()
        positions = {name: 0 for name in _FAMILIES}
        while len(selected) < self._cfg.max_preview_images:
            progressed = False
            for name in _FAMILIES:
                members = self._families[name]
                index = positions[name]
                while index < len(members) and members[index][-1] in seen:
                    index += 1
                positions[name] = index
                if index < len(members) and len(selected) < self._cfg.max_preview_images:
                    key_id = members[index][-1]
                    positions[name] = index + 1
                    seen.add(key_id)
                    selected.append(key_id)
                    progressed = True
            if not progressed:
                break
        return selected

    def _memberships(self) -> dict[str, list[str]]:
        memberships: dict[str, list[str]] = {}
        for name in _FAMILIES:
            for entry in self._families[name]:
                memberships.setdefault(entry[-1], []).append(name)
        return memberships

    def _build_refs(self, kept: list[str]) -> Result[tuple[ArtifactRef, ...], str]:
        refs: list[ArtifactRef] = []

        def add_ref(
            path: Path,
            bundle_path: str,
            kind: ArtifactKind,
            fmt: str,
            rows: int | None,
            population: str,
        ) -> str | None:
            ref = artifact_file_ref(
                path,
                bundle_path=bundle_path,
                kind=kind,
                format=fmt,
                rows=rows,
                population=population,
            )
            if isinstance(ref, Err):
                return ref.error
            refs.append(ref.value)
            return None

        problem = add_ref(
            self._out / ROWS_FILENAME,
            ROWS_FILENAME,
            "TABLE",
            "jsonl",
            len(self._seen),
            "evaluated_rows",
        )
        if problem is None:
            problem = add_ref(
                self._out / METADATA_FILENAME,
                METADATA_FILENAME,
                "CONFIG",
                "json",
                None,
                "capture_metadata",
            )
        for key_id in kept:
            if problem is not None:
                break
            problem = add_ref(
                self._out / PREVIEWS_DIR / f"{key_id}.npz",
                f"{PREVIEWS_DIR}/{key_id}.npz",
                "VISUAL",
                "npz",
                1,
                "selected_previews",
            )
        for bundle_path in self._full_written:
            if problem is not None:
                break
            problem = add_ref(
                self._out.joinpath(*bundle_path.split("/")),
                bundle_path,
                "PREDICTIONS",
                "npz",
                1,
                "full_predictions",
            )
        if problem is not None:
            return Err(problem)
        return Ok(tuple(refs))

    def references(self) -> tuple[ArtifactRef, ...]:
        """Artifact references; empty until close succeeds."""
        return self._refs
