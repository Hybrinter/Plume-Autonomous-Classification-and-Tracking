"""Exhaustive, identity-bound evaluation with canonical train populations.

Contains:
  - evaluate_split: the public ``Result`` boundary.
  - canonical_array: inverse dihedral view for channel-first evidence arrays.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from flight.libs.types import Err, Ok, Result
from torch import nn

from tools.ml_models.analysis.capture import CaptureRow, CaptureSink
from tools.ml_models.analysis.config import EvaluationConfig, ScoreConfig
from tools.ml_models.analysis.contracts import (
    CurveEvidence,
    MetricSupport,
    MetricValue,
    NamedCount,
    SampleKey,
    SplitEvidence,
    StratumEvidence,
)
from tools.ml_models.analysis.metrics.classifier import score_classifier
from tools.ml_models.analysis.metrics.segmentation import (
    SegmentationAccumulator,
    SegmentationRow,
    score_segmentation_image,
)
from tools.ml_models.analysis.metrics.spatial import (
    SpatialRow,
    aggregate_spatial,
    score_spatial,
)
from tools.ml_models.dataset.augment import ELEMENT_NAMES, apply_dihedral, legal_elements
from tools.ml_models.dataset.loader import ShardDataset
from tools.ml_models.dataset.manifest import DatasetManifest, load_manifest, shard_dir
from tools.ml_models.dataset.store import read_gsd, read_labels, read_rows
from tools.ml_models.train.losses import PlumeLoss


def canonical_array(array: npt.NDArray[np.float32], element: str) -> npt.NDArray[np.float32]:
    """Invert a known dihedral element on one channel-first spatial array."""
    inverse = "rot270" if element == "rot90" else "rot90" if element == "rot270" else element
    return np.asarray(apply_dihedral(array, inverse), dtype=np.float32)


def _scalar_metric(name: str, value: float | None, n: int, reason: str) -> MetricValue:
    return MetricValue(
        name=name,
        value=value,
        status="AVAILABLE" if value is not None else "UNAVAILABLE",
        reason=None if value is not None else reason,
        aggregation="equal_image_mean",
        support=MetricSupport(unit="IMAGE", n=n),
    )


@dataclass(slots=True)
class _Cohort:
    logits: list[float] = field(default_factory=list)
    labels: list[float] = field(default_factory=list)
    segmentation: SegmentationAccumulator = field(default_factory=SegmentationAccumulator)
    spatial: list[SpatialRow] = field(default_factory=list)
    objectives: dict[str, list[float]] = field(default_factory=dict)
    groups: set[str] = field(default_factory=set)
    n: int = 0

    def add(
        self,
        label: float,
        logit: float | None,
        segmentation: SegmentationRow | None,
        objectives: dict[str, float],
        group_id: str,
        spatial: SpatialRow | None = None,
    ) -> Result[None, str]:
        if segmentation is not None:
            if spatial is None:
                return Err("segmentor cohort rows require spatial diagnostics")
            added = self.segmentation.add(segmentation)
            if isinstance(added, Err):
                return added
            self.spatial.append(spatial)
        elif spatial is not None:
            return Err("classifier cohort rows carry no spatial diagnostics")
        if logit is not None:
            self.logits.append(logit)
            self.labels.append(label)
        for name, value in objectives.items():
            self.objectives.setdefault(name, []).append(value)
        self.groups.add(group_id)
        self.n += 1
        return Ok(None)

    def evidence(
        self, cfg: EvaluationConfig
    ) -> Result[tuple[tuple[MetricValue, ...], tuple[CurveEvidence, ...], MetricSupport], str]:
        measured = (
            score_classifier(self.logits, self.labels, cfg.score)
            if cfg.kind == "classifier"
            else self.segmentation.result()
        )
        if isinstance(measured, Err):
            return measured
        metrics = measured.value.metrics
        curves = measured.value.curves
        if cfg.kind == "segmentor":
            spatial = aggregate_spatial(tuple(self.spatial))
            if isinstance(spatial, Err):
                return spatial
            metrics += spatial.value.metrics
            curves += spatial.value.curves
        support = replace(
            measured.value.support,
            counts=measured.value.support.counts
            + (NamedCount(name="n_groups", value=len(self.groups)),),
        )
        objectives = tuple(
            _scalar_metric(
                name,
                math.fsum(value / len(values) for value in values) if values else None,
                len(values),
                "objective was not supplied or this component is inactive",
            )
            for name in ("objective_loss", "objective_bce", "objective_focal", "objective_dice")
            for values in (self.objectives.get(name, []),)
        )
        return Ok((metrics + objectives, curves, support))


def _objective_rows(
    objective: PlumeLoss | None, logits: torch.Tensor, targets: torch.Tensor
) -> list[dict[str, float]]:
    records: list[dict[str, float]] = [{} for _ in range(logits.shape[0])]
    if objective is None:
        return records
    components = objective.per_sample_components(logits, targets)
    for name, values in (
        ("objective_loss", components.total),
        ("objective_bce", components.bce),
        ("objective_focal", components.focal),
        ("objective_dice", components.dice),
    ):
        if values is None:
            continue
        if values.shape != (len(records),) or not bool(torch.isfinite(values).all()):
            raise ValueError("configured objective produced nonfinite or misaligned losses")
        for record, value in zip(records, values.detach().cpu().tolist(), strict=True):
            record[name] = float(value)
    return records


def _row_metrics(
    logits: npt.NDArray[np.float32],
    label: float,
    segmentation: SegmentationRow | None,
    objective: dict[str, float],
    cfg: ScoreConfig,
    spatial: SpatialRow | None = None,
) -> tuple[tuple[MetricValue, ...], float, bool, bool]:
    if segmentation is None:
        measured = score_classifier(logits, [label], cfg)
        if isinstance(measured, Err):
            raise ValueError(measured.error)
        selected = measured.value
        metrics = tuple(
            metric
            for metric in selected.metrics
            if metric.name in ("binary_cross_entropy", "brier_score")
        ) + (
            _scalar_metric("logit", float(logits[0]), 1, ""),
            _scalar_metric("probability", selected.probabilities[0], 1, ""),
        )
        loss = selected.losses[0]
        if loss is None:
            raise ValueError("finite classifier logits require finite unweighted BCE")
        fp, fn = selected.counts.fp > 0, selected.counts.fn > 0
        failure = loss
    else:
        metrics = tuple(
            _scalar_metric(name, value, 1, "no eligible pixels")
            for name, value in (
                ("foreground_iou", segmentation.iou),
                ("foreground_dice", segmentation.dice),
                ("binary_cross_entropy", segmentation.bce),
                ("brier_score", segmentation.brier),
                ("target_area_px", float(segmentation.target_area_px)),
                ("predicted_area_px", float(segmentation.predicted_area_px)),
                ("target_area_m2", segmentation.target_area_m2),
                ("predicted_area_m2", segmentation.predicted_area_m2),
                ("true_positive_pixels", float(segmentation.tp)),
                ("false_positive_pixels", float(segmentation.fp)),
                ("true_negative_pixels", float(segmentation.tn)),
                ("false_negative_pixels", float(segmentation.fn)),
                ("predicted_blobs", float(segmentation.n_predicted_blobs)),
                ("foreground_brier_score", segmentation.foreground_brier),
                ("background_brier_score", segmentation.background_brier),
                ("foreground_probability_residual", segmentation.foreground_probability_residual),
                ("background_probability_residual", segmentation.background_probability_residual),
            )
        )
        if spatial is None:
            raise ValueError("segmentor rows require spatial diagnostics")
        measured_spatial = aggregate_spatial((spatial,))
        if isinstance(measured_spatial, Err):
            raise ValueError(measured_spatial.error)
        metrics += measured_spatial.value.metrics
        fp = segmentation.verified_empty and segmentation.predicted_area_px > 0
        fn = segmentation.target_area_px > 0 and segmentation.predicted_area_px == 0
        failure = 1.0 - segmentation.iou
    metrics += tuple(_scalar_metric(name, value, 1, "") for name, value in objective.items())
    return metrics, failure, fp, fn


def _evaluate(
    model: nn.Module,
    dataset: Path,
    manifest: DatasetManifest,
    cfg: EvaluationConfig,
    objective: PlumeLoss | None,
    capture: CaptureSink | None,
) -> Result[SplitEvidence, str]:
    manifest_path = dataset / "dataset.json"
    if load_manifest(manifest_path) != manifest:
        return Err("supplied dataset manifest disagrees with verified dataset identity")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    selected = tuple(
        shard for shard in manifest.shards if shard.task == cfg.kind and shard.split == cfg.split
    )
    if not selected:
        return Err(f"dataset has no {cfg.kind}/{cfg.split} rows")
    if len({(shard.height, shard.width) for shard in selected}) != len(selected):
        return Err("dataset manifest repeats a spatial shard")
    total = _Cohort()
    bins: dict[str, _Cohort] = {}
    skipped = 0
    inverted = 0
    for count in selected:
        directory = shard_dir(dataset, cfg.kind, cfg.split, count.height, count.width)
        shard = ShardDataset(
            directory, manifest.gsd_reference_m, cfg.kind, channels=len(manifest.band_names)
        )
        rows, labels, gsd = read_rows(directory), read_labels(directory), read_gsd(directory)
        if len(shard) != count.n or len(rows) != count.n:
            return Err("evaluation shard row count disagrees with manifest")
        if not np.isin(labels, (0.0, 1.0)).all():
            return Err("evaluation labels must be exactly binary")
        if int(np.count_nonzero(labels)) != count.n_positive:
            return Err("evaluation positive count disagrees with manifest")
        if not np.isfinite(gsd).all() or np.any(gsd <= 0):
            return Err("evaluation GSD must be finite positive metres")
        if any(row.element not in legal_elements(count.height, count.width) for row in rows):
            return Err("evaluation row has an unknown or non-axis-preserving augmentation element")
        if cfg.split != "train" and any(row.element != "id" for row in rows):
            return Err("validation and test require identity views")
        identities = [
            (row.tile_id, row.bin_id, float(gsd[i, 0]), float(gsd[i, 1]))
            for i, row in enumerate(rows)
        ]
        choices: dict[tuple[str, str, float, float], list[int]] = {}
        for index, identity in enumerate(identities):
            choices.setdefault(identity, []).append(index)
        if any(
            len({rows[index].element for index in variants}) != len(variants)
            for variants in choices.values()
        ):
            return Err("evaluation repeats a tile/GSD augmentation identity")
        indices = sorted(
            min(variants, key=lambda index: ELEMENT_NAMES.index(rows[index].element))
            for variants in choices.values()
        )
        skipped += count.n - len(indices)
        inverted += sum(rows[index].element != "id" for index in indices)
        for offset in range(0, len(indices), cfg.batch_size):
            batch_indices = indices[offset : offset + cfg.batch_size]
            samples = []
            for index in batch_indices:
                image, encoded, target = shard[index]
                if rows[index].element != "id":
                    image = torch.from_numpy(canonical_array(image.numpy(), rows[index].element))
                    if cfg.kind == "segmentor":
                        target = torch.from_numpy(
                            canonical_array(target.numpy(), rows[index].element)
                        )
                samples.append((image, encoded, target))
            images, encoded_gsd, targets = (
                torch.stack([sample[axis] for sample in samples]).to(cfg.device)
                for axis in range(3)
            )
            if not bool(((targets == 0) | (targets == 1)).all()):
                return Err("evaluation targets must be exactly binary")
            if tuple(images.shape[2:]) != (count.height, count.width):
                return Err("evaluation spatial shape disagrees with manifest")
            outputs = model(images, encoded_gsd)
            if not isinstance(outputs, torch.Tensor):
                return Err("model must return one logit tensor")
            if outputs.shape != targets.shape or outputs.dtype != torch.float32:
                return Err("model logit shape or dtype disagrees with evaluation targets")
            if not bool(torch.isfinite(outputs).all()):
                return Err("model logits must be finite")
            objectives = _objective_rows(objective, outputs, targets)
            logits_array = outputs.detach().cpu().numpy().astype(np.float32)
            targets_array = targets.detach().cpu().numpy().astype(np.float32)
            images_array = images.detach().cpu().numpy().astype(np.float32)
            for position, index in enumerate(batch_indices):
                row = rows[index]
                label = float(labels[index, 0])
                raw_gsd = (float(gsd[index, 0]), float(gsd[index, 1]))
                logits = logits_array[position]
                segmentation = None
                spatial = None
                if cfg.kind == "segmentor":
                    measured = score_segmentation_image(
                        logits,
                        targets_array[position],
                        label=label,
                        verified_empty=label == 0.0 and not bool(targets_array[position].any()),
                        gsd=raw_gsd,
                        cfg=cfg.score,
                    )
                    if isinstance(measured, Err):
                        return measured
                    segmentation = measured.value
                    spatial_row = score_spatial(
                        logits,
                        targets_array[position],
                        gsd=raw_gsd,
                        cfg=cfg.score,
                    )
                    if isinstance(spatial_row, Err):
                        return spatial_row
                    spatial = spatial_row.value
                logit = float(logits[0]) if cfg.kind == "classifier" else None
                for cohort in (total, bins.setdefault(row.bin_id, _Cohort())):
                    added = cohort.add(
                        label, logit, segmentation, objectives[position], row.group_id, spatial
                    )
                    if isinstance(added, Err):
                        return added
                if capture is not None:
                    metrics, failure, fp, fn = _row_metrics(
                        logits, label, segmentation, objectives[position], cfg.score, spatial
                    )
                    captured = capture.add(
                        CaptureRow(
                            key=SampleKey(
                                dataset_hash=manifest.dataset_hash,
                                task=cfg.kind,
                                split=cfg.split,
                                spatial_shard=(count.height, count.width),
                                row_index=index,
                                tile_id=row.tile_id,
                                element=row.element,
                            ),
                            group_id=row.group_id,
                            bin_id=row.bin_id,
                            label=label,
                            gsd_m=raw_gsd,
                            metrics=metrics,
                            failure_score=failure,
                            false_positive=fp,
                            false_negative=fn,
                            spatial=spatial,
                        ),
                        image=images_array[position],
                        target=targets_array[position],
                        logits=logits,
                    )
                    if isinstance(captured, Err):
                        return captured
    scored = total.evidence(cfg)
    if isinstance(scored, Err):
        return scored
    strata: list[StratumEvidence] = []
    for bin_id, cohort in sorted(bins.items()):
        scored_bin = cohort.evidence(cfg)
        if isinstance(scored_bin, Err):
            return scored_bin
        metrics, _, support = scored_bin.value
        strata.append(
            StratumEvidence(name="gsd_bin", value=bin_id or None, metrics=metrics, support=support)
        )
    metrics, curves, support = scored.value
    warnings: tuple[str, ...] = (
        (f"Canonical train evaluation excluded {skipped} augmentation copies.",) if skipped else ()
    )
    if inverted:
        warnings += (f"Canonical train evaluation inverted {inverted} non-identity views.",)
    warnings += tuple(
        f"{metric.name}: {metric.reason}" for metric in metrics if metric.status == "UNAVAILABLE"
    )
    return Ok(
        SplitEvidence(
            task=cfg.kind,
            split=cfg.split,
            dataset_hash=manifest.dataset_hash,
            dataset_manifest_hash=manifest_hash,
            metrics=metrics,
            curves=curves,
            support=support,
            warnings=warnings,
            strata=tuple(strata),
        )
    )


def evaluate_split(
    model: nn.Module,
    dataset: Path,
    manifest: DatasetManifest,
    cfg: EvaluationConfig,
    objective: PlumeLoss | None = None,
    capture: CaptureSink | None = None,
) -> Result[SplitEvidence, str]:
    """Evaluate every eligible row and restore every module's incoming mode.

    Args:
        model: Conditioned model called as ``model(image, encoded_gsd)``.
        dataset: Finished dataset directory, verified before inference.
        manifest: Parsed dataset manifest, required to match the verified file.
        cfg: Evaluation inputs.
        objective: Optional configured objective for separate loss records.
        capture: Optional sink, closed on both success and failure.

    Returns:
        Result[SplitEvidence, str]: Measured evidence, or an explicit error.
    """
    modes = tuple((module, module.training) for module in model.modules())
    result: Result[SplitEvidence, str]
    try:
        model.eval()
        with torch.inference_mode():
            result = _evaluate(model, Path(dataset), manifest, cfg, objective, capture)
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError) as exc:
        result = Err(f"split evaluation failed: {exc}")
    finally:
        for module, training in modes:
            module.training = training
    if capture is not None:
        if isinstance(result, Err):
            try:
                aborted = capture.abort(result.error)
            except (OSError, ValueError, RuntimeError) as exc:
                aborted = Err(str(exc))
            if isinstance(aborted, Err):
                result = Err(f"{result.error}; capture abort failed: {aborted.error}")
        try:
            closed = capture.close()
        except (OSError, ValueError, RuntimeError) as exc:
            closed = Err(f"capture close failed: {exc}")
        if isinstance(closed, Err):
            prior = f"{result.error}; " if isinstance(result, Err) else ""
            return Err(f"{prior}capture close failed: {closed.error}")
        if isinstance(result, Ok):
            result = Ok(replace(result.value, artifacts=capture.references()))
    return result
