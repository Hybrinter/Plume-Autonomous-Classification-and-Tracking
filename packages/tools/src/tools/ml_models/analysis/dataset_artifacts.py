"""Dataset evidence bundle persistence and assembly.

The publisher freezes a caller-supplied ``DatasetMeasurement`` into an
exclusive checksummed bundle: canonical ``measurements.json``, typed
scalar tables, a table-schema manifest, and the resolved ``config.toml``.
It never traverses, reloads, or recomputes dataset inputs; every artifact
is staged in a private temporary directory and read back into
``BundleFile`` bytes before ``publish_bundle`` reserves the destination.

Contains:
  - SAMPLES_SCHEMA, COMPONENTS_SCHEMA, COVERAGE_SCHEMA, TABLE_SCHEMAS:
    exact column contracts for the published tables.
  - code_identity: git revision and dirty state for this source tree.
  - publish_dataset_measurement: the ``Result`` publication boundary.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

from flight.libs.types import Err, Result

from tools.ml_models.analysis.artifacts import (
    BundleFile,
    ColumnSpec,
    Scalar,
    TableSchema,
    artifact_file_ref,
    artifact_ref,
    publish_bundle,
    write_table,
)
from tools.ml_models.analysis.config import DatasetAnalysisConfig, write_config
from tools.ml_models.analysis.contracts import (
    ArtifactRef,
    AvailabilityRecord,
    CodeIdentity,
)

if TYPE_CHECKING:
    from tools.ml_models.analysis.dataset import DatasetMeasurement

_GIT_TIMEOUT_S = 10
_MEASUREMENTS_PATH = "measurements.json"
_CONFIG_PATH = "config.toml"
_SCHEMAS_PATH = "tables/schema.json"

SAMPLES_SCHEMA = TableSchema(
    columns=(
        ColumnSpec(name="variant_id", dtype="STRING"),
        ColumnSpec(name="dataset_hash", dtype="STRING"),
        ColumnSpec(name="task", dtype="STRING"),
        ColumnSpec(name="split", dtype="STRING"),
        ColumnSpec(name="height", dtype="INT64"),
        ColumnSpec(name="width", dtype="INT64"),
        ColumnSpec(name="row_index", dtype="INT64"),
        ColumnSpec(name="tile_id", dtype="STRING"),
        ColumnSpec(name="element", dtype="STRING"),
        ColumnSpec(name="group_id", dtype="STRING"),
        ColumnSpec(name="bin_id", dtype="STRING"),
        ColumnSpec(name="label", dtype="INT64"),
        ColumnSpec(name="lateral_gsd_m", dtype="FLOAT64"),
        ColumnSpec(name="along_gsd_m", dtype="FLOAT64"),
        ColumnSpec(name="gsd_anisotropy", dtype="FLOAT64"),
        ColumnSpec(name="tile_area_m2", dtype="FLOAT64"),
        ColumnSpec(name="gsd_nominal", dtype="BOOLEAN"),
        ColumnSpec(name="stored_rows", dtype="INT64"),
        ColumnSpec(name="tasks_json", dtype="STRING"),
        ColumnSpec(name="image_sha256", dtype="STRING"),
        ColumnSpec(name="observation_id", dtype="STRING", nullable=True),
        ColumnSpec(name="acquired_at_utc", dtype="STRING", nullable=True),
        ColumnSpec(name="conditions_json", dtype="STRING"),
        ColumnSpec(name="annotation_source", dtype="STRING", nullable=True),
        ColumnSpec(name="annotation_version", dtype="STRING", nullable=True),
        ColumnSpec(name="source_annotation_state", dtype="STRING"),
        ColumnSpec(name="prepared_mask_state", dtype="STRING"),
        ColumnSpec(name="mask_area_px", dtype="INT64", nullable=True),
        ColumnSpec(name="mask_area_m2", dtype="FLOAT64", nullable=True),
        ColumnSpec(name="mask_area_fraction", dtype="FLOAT64", nullable=True),
        ColumnSpec(name="mask_components", dtype="INT64", nullable=True),
        ColumnSpec(name="mask_border_touching", dtype="BOOLEAN", nullable=True),
        ColumnSpec(name="mask_task", dtype="STRING", nullable=True),
        ColumnSpec(name="mask_row_index", dtype="INT64", nullable=True),
        ColumnSpec(name="mask_element", dtype="STRING", nullable=True),
    )
)

COMPONENTS_SCHEMA = TableSchema(
    columns=(
        ColumnSpec(name="variant_id", dtype="STRING"),
        ColumnSpec(name="component_index", dtype="INT64"),
        ColumnSpec(name="area_px", dtype="INT64"),
        ColumnSpec(name="area_m2", dtype="FLOAT64"),
    )
)

COVERAGE_SCHEMA = TableSchema(
    columns=(
        ColumnSpec(name="population", dtype="STRING"),
        ColumnSpec(name="split", dtype="STRING", nullable=True),
        ColumnSpec(name="field", dtype="STRING"),
        ColumnSpec(name="value", dtype="STRING", nullable=True),
        ColumnSpec(name="n", dtype="INT64"),
        ColumnSpec(name="total", dtype="INT64"),
    )
)

TABLE_SCHEMAS: dict[str, TableSchema] = {
    "tables/samples.parquet": SAMPLES_SCHEMA,
    "tables/components.parquet": COMPONENTS_SCHEMA,
    "tables/coverage.csv": COVERAGE_SCHEMA,
}


def code_identity() -> CodeIdentity:
    """Capture git revision and dirty state for the worktree holding this file.

    The commands run inside the source module's directory, not the caller's
    working directory. ``diff_hash`` stays None; unavailability is explicit
    and never fabricated. No environment or credentials are read.
    """
    cwd = Path(__file__).resolve().parent
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "status", "--porcelain", "--untracked-files=normal"),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        ).stdout
    except (OSError, UnicodeError, subprocess.SubprocessError) as exc:
        return CodeIdentity(
            revision=None,
            dirty=None,
            reason=f"git provenance unavailable: {exc}",
        )
    return CodeIdentity(revision=revision, dirty=bool(status.strip()))


def _sample_rows(measured: DatasetMeasurement) -> tuple[dict[str, Scalar], ...]:
    """Flatten frozen samples to scalar table rows; no dataset reads."""
    rows: list[dict[str, Scalar]] = []
    for sample in measured.samples:
        key = sample.key
        row = sample.row
        metadata = row.metadata
        mask = sample.mask
        height, width = key.spatial_shard
        rows.append(
            {
                "variant_id": sample.variant_id,
                "dataset_hash": key.dataset_hash,
                "task": key.task,
                "split": key.split,
                "height": height,
                "width": width,
                "row_index": key.row_index,
                "tile_id": key.tile_id,
                "element": key.element,
                "group_id": row.group_id,
                "bin_id": row.bin_id,
                "label": sample.label,
                "lateral_gsd_m": sample.gsd_m[0],
                "along_gsd_m": sample.gsd_m[1],
                "gsd_anisotropy": sample.gsd_anisotropy,
                "tile_area_m2": sample.tile_area_m2,
                "gsd_nominal": row.gsd_nominal,
                "stored_rows": sample.stored_rows,
                "tasks_json": json.dumps(list(sample.tasks), separators=(",", ":")),
                "image_sha256": sample.image_sha256,
                "observation_id": metadata.observation_id,
                "acquired_at_utc": metadata.acquired_at_utc,
                "conditions_json": json.dumps(
                    [asdict(tag) for tag in metadata.conditions], separators=(",", ":")
                ),
                "annotation_source": metadata.annotation_source,
                "annotation_version": metadata.annotation_version,
                "source_annotation_state": metadata.source_annotation_state,
                "prepared_mask_state": row.prepared_mask_state,
                "mask_area_px": mask.area_px if mask is not None else None,
                "mask_area_m2": mask.area_m2 if mask is not None else None,
                "mask_area_fraction": mask.area_fraction if mask is not None else None,
                "mask_components": mask.n_components if mask is not None else None,
                "mask_border_touching": mask.border_touching if mask is not None else None,
                "mask_task": sample.mask_key.task if sample.mask_key is not None else None,
                "mask_row_index": (
                    sample.mask_key.row_index if sample.mask_key is not None else None
                ),
                "mask_element": (sample.mask_key.element if sample.mask_key is not None else None),
            }
        )
    return tuple(rows)


def _component_rows(measured: DatasetMeasurement) -> tuple[dict[str, Scalar], ...]:
    """Flatten frozen component records to scalar table rows."""
    return tuple(
        {
            "variant_id": component.variant_id,
            "component_index": component.component_index,
            "area_px": component.area_px,
            "area_m2": component.area_m2,
        }
        for component in measured.components
    )


def _coverage_rows(measured: DatasetMeasurement) -> tuple[dict[str, Scalar], ...]:
    """Flatten frozen coverage records to scalar table rows."""
    return tuple(
        {
            "population": record.population,
            "split": record.split,
            "field": record.field,
            "value": record.value,
            "n": record.n,
            "total": record.total,
        }
        for record in measured.coverage
    )


def publish_dataset_measurement(
    measured: DatasetMeasurement,
    cfg: DatasetAnalysisConfig,
    *,
    extra_files: tuple[BundleFile, ...] = (),
    extra_refs: tuple[ArtifactRef, ...] = (),
    extra_outputs: tuple[AvailabilityRecord, ...] = (),
    code: CodeIdentity | None = None,
) -> Result[Path, str]:
    """Stage, checksum, and exclusively publish one dataset evidence bundle.

    All codecs run inside a private temporary directory before the output is
    reserved, so a codec failure leaves no bundle behind. ``publish_bundle``
    retains its ``.incomplete`` marker when a write fails; existing outputs
    are never overwritten. ``extra_files``/``extra_refs`` carry pre-rendered
    artifacts bound by the caller; ``extra_outputs`` merge into the summary by
    name (duplicates within the batch are rejected; a supplied name replaces
    the base record, e.g. flipping ``figures`` to ``AVAILABLE``). ``code``
    defaults to ``code_identity()``.
    """
    out = Path(cfg.out)
    try:
        with tempfile.TemporaryDirectory(prefix=".dataset-analysis-") as staging:
            staged = Path(staging)
            config_file = staged / _CONFIG_PATH
            written_config = write_config(config_file, cfg)
            if isinstance(written_config, Err):
                return written_config
            config_ref = artifact_file_ref(
                config_file, bundle_path=_CONFIG_PATH, kind="CONFIG", format="toml"
            )
            if isinstance(config_ref, Err):
                return config_ref
            measurements_bytes = json.dumps(
                asdict(measured), sort_keys=True, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
            measurements_ref = artifact_ref(
                _MEASUREMENTS_PATH, measurements_bytes, kind="REFERENCE", format="json"
            )
            if isinstance(measurements_ref, Err):
                return measurements_ref
            table_rows = {
                "tables/samples.parquet": _sample_rows(measured),
                "tables/components.parquet": _component_rows(measured),
                "tables/coverage.csv": _coverage_rows(measured),
            }
            files: list[BundleFile] = [
                BundleFile(path=_CONFIG_PATH, data=config_file.read_bytes()),
                BundleFile(path=_MEASUREMENTS_PATH, data=measurements_bytes),
            ]
            refs: list[ArtifactRef] = [config_ref.value, measurements_ref.value]
            for rel in sorted(TABLE_SCHEMAS):
                table_file = staged.joinpath(*rel.split("/"))
                table_file.parent.mkdir(parents=True, exist_ok=True)
                table_ref = write_table(
                    table_file, table_rows[rel], TABLE_SCHEMAS[rel], bundle_path=rel
                )
                if isinstance(table_ref, Err):
                    return table_ref
                refs.append(table_ref.value)
                files.append(BundleFile(path=rel, data=table_file.read_bytes()))
            schemas_bytes = json.dumps(
                {rel: asdict(TABLE_SCHEMAS[rel]) for rel in sorted(TABLE_SCHEMAS)},
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            schemas_ref = artifact_ref(
                _SCHEMAS_PATH, schemas_bytes, kind="REFERENCE", format="json"
            )
            if isinstance(schemas_ref, Err):
                return schemas_ref
            refs.append(schemas_ref.value)
            files.append(BundleFile(path=_SCHEMAS_PATH, data=schemas_bytes))
            from tools.ml_models.analysis.dataset import dataset_summary

            summary = dataset_summary(
                measured,
                cfg,
                code if code is not None else code_identity(),
                tuple(refs) + tuple(extra_refs),
            )
            if extra_outputs:
                merged = {record.name: record for record in summary.outputs}
                seen: set[str] = set()
                for record in extra_outputs:
                    if record.name in seen:
                        return Err(f"duplicate extra output name {record.name!r}")
                    if record.name in merged and record.name not in ("figures", "visuals"):
                        return Err(f"cannot replace measured availability {record.name!r}")
                    seen.add(record.name)
                    merged[record.name] = record
                summary = replace(summary, outputs=tuple(merged.values()))
    except (OSError, TypeError, ValueError, OverflowError, RuntimeError) as exc:
        return Err(f"dataset publication failed before reservation: {exc}")
    try:
        return publish_bundle(
            out,
            summary,
            tuple(files) + tuple(extra_files),
            dataset_root=Path(cfg.dataset),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        return Err(f"dataset publication failed: {exc}")
