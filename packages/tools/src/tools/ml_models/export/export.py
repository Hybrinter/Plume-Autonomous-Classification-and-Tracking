"""Export a trained conditioned checkpoint as a two-input ONNX artifact.

Traces ``model(image, gsd)`` at the 193x258 flight tile with a dynamic batch
(fixed flight spatial dims by default), verifies the declared graph metadata,
records actual training GSD coverage in the JSON sidecar, and publishes both
files through private sibling temporaries linked into place. Existing outputs
are never overwritten — including files created concurrently between the
precheck and publish — and nothing is copied to deployment locations.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from flight.libs.config import InferenceConfig
from flight.libs.types import Err, Ok, Result

from tools.ml_models.export.contract import (
    CONDITIONING_ID,
    GSD_ENCODING,
    coverage_ok,
    required_gsd_coverage,
)
from tools.ml_models.export.manifest import (
    BAND_NAMES,
    TILE_HW,
    ModelManifest,
    sidecar_path,
    write_manifest,
)

_TRACE_CHANNELS = 3


@dataclass(frozen=True, slots=True)
class ExportConfig:
    """Export inputs; ``checkpoint_path`` and ``output_path`` are required."""

    checkpoint_path: str
    output_path: str
    opset: int = 17
    dynamic_spatial: bool = False
    allow_partial_gsd: bool = False


def _dim_of(dim: object) -> int | str | None:
    """Read one onnx ``TensorShapeProto.Dimension`` as int, symbol, or None."""
    proto = cast(Any, dim)
    if proto.HasField("dim_value"):
        return int(proto.dim_value)
    if proto.HasField("dim_param"):
        return str(proto.dim_param)
    return None


def _declared_shape(tensor: object) -> tuple[int | None, ...]:
    """Normalize an onnx ``TypeProto`` tensor shape to ints and None."""
    proto = cast(Any, tensor)
    dims: list[int | None] = []
    for dim in proto.type.tensor_type.shape.dim:
        value = _dim_of(dim)
        dims.append(value if isinstance(value, int) else None)
    return tuple(dims)


def _verify_graph(path: Path, kind: str) -> tuple[tuple[int | None, ...], ...]:
    """Load ``path`` with onnx and return (image, gsd, output) declared shapes.

    Raises ValueError when the graph metadata violates the contract.
    """
    import onnx

    from tools.ml_models.export.contract import verify_conditioned_shapes

    proto = onnx.load(str(path))
    input_names = [value.name for value in proto.graph.input]
    if len(input_names) != len(set(input_names)):
        raise ValueError("exported graph declares duplicate input names")
    inputs = {value.name: value for value in proto.graph.input}
    outputs = list(proto.graph.output)
    if len(inputs) != 2 or set(inputs) != {"image", "gsd"}:
        raise ValueError("exported graph inputs must be exactly {'image', 'gsd'}")
    if len(outputs) != 1:
        raise ValueError("exported graph must declare exactly one output")
    float_type = onnx.TensorProto.FLOAT
    for value in (*inputs.values(), *outputs):
        if value.type.tensor_type.elem_type != float_type:
            raise ValueError(f"{value.name} must be tensor(float)")
    image = _declared_shape(inputs["image"])
    gsd = _declared_shape(inputs["gsd"])
    output = _declared_shape(outputs[0])
    batch_dims = (
        _dim_of(inputs["image"].type.tensor_type.shape.dim[0]),
        _dim_of(inputs["gsd"].type.tensor_type.shape.dim[0]),
        _dim_of(outputs[0].type.tensor_type.shape.dim[0]),
    )
    if all(isinstance(dim, str) for dim in batch_dims):
        if len(set(batch_dims)) != 1:
            raise ValueError("batch dims must share one named symbol")
    elif not all(dim is None for dim in batch_dims):
        raise ValueError("batch dims must be dynamic on every I/O tensor")
    # The flight validator's tile argument means an exact spatial extent.
    # Export accepts dynamic spatial axes, so validate those separately and
    # ask the shared validator for the batch/channel/GSD/output contract.
    if any(dim not in (None, expected) for dim, expected in zip(image[2:], TILE_HW, strict=True)):
        raise ValueError("image spatial dimensions must be dynamic or match the flight tile")
    if kind == "segmentor":
        if any(
            dim not in (None, expected) for dim, expected in zip(output[2:], TILE_HW, strict=True)
        ):
            raise ValueError(
                "segmentor output spatial dimensions must be dynamic or match the flight tile"
            )
        if any(
            image_dim is not None and output_dim is not None and image_dim != output_dim
            for image_dim, output_dim in zip(image[2:], output[2:], strict=True)
        ):
            raise ValueError("segmentor output spatial dimensions must match image dimensions")
    result = verify_conditioned_shapes(image, gsd, output, _TRACE_CHANNELS, None, kind)
    if isinstance(result, Err):
        raise ValueError(f"exported graph violates the flight contract ({result.error.value})")
    return image, gsd, output


def _refine_shape_metadata(path: Path) -> None:
    """Propagate truthful Resize output dimensions into ONNX value metadata.

    PyTorch's legacy exporter can leave the segmentor's one-channel output and
    fixed-tile spatial dimensions symbolic even when they are derivable from
    the graph. ONNX data propagation resolves those dimensions for fixed tiles
    and preserves symbols for dynamic research exports.
    """
    import onnx

    try:
        proto = onnx.load(str(path))
        refined = onnx.shape_inference.infer_shapes(proto, data_prop=True)
        onnx.checker.check_model(refined)
        onnx.save(refined, str(path))
    except Exception as exc:  # noqa: BLE001 - surface SDK graph errors as export errors.
        raise ValueError(f"ONNX shape inference failed: {exc}") from exc


def _private_sibling(dest: Path) -> Path:
    """Create a uniquely named sibling temp file and return its path."""
    descriptor, name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".partial", dir=dest.parent)
    os.close(descriptor)
    return Path(name)


def _publish(tmp: Path, dest: Path) -> None:
    """Publish ``tmp`` at ``dest`` exclusively; the temp is ours to keep.

    ``os.link`` fails when ``dest`` exists — including files created after
    the earlier precheck — so concurrent outputs are never overwritten. The
    temporary is removed by the caller's cleanup so rollback can compare it
    to the just-linked destination.
    """
    os.link(tmp, dest)


def _export(cfg: ExportConfig) -> Path:
    import torch

    from tools.ml_models.arch.registry import build

    artifact = Path(cfg.output_path)
    sidecar = sidecar_path(artifact)
    if artifact.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite {artifact} or {sidecar}")
    checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["conditioning"] != CONDITIONING_ID:
        raise ValueError(
            f"checkpoint conditioning {checkpoint['conditioning']!r} is not {CONDITIONING_ID!r}; "
            "only conditioned pactnet/dilatenet checkpoints export"
        )
    kind = cast("Literal['classifier', 'segmentor']", str(checkpoint["kind"]))
    if kind not in ("classifier", "segmentor"):
        raise ValueError(f"unknown checkpoint kind {kind!r}")
    arch = str(checkpoint["arch"])
    provenance = checkpoint["provenance"]
    band_names = tuple(str(band) for band in provenance["band_names"])
    if provenance["norm"] != "unit":
        raise ValueError(f"provenance norm must be 'unit'; got {provenance['norm']!r}")
    if band_names != BAND_NAMES:
        raise ValueError(f"provenance band_names must be {BAND_NAMES}; got {band_names}")
    gsd_reference = float(provenance["gsd_reference_m"])
    flight_reference = InferenceConfig().gsd_reference_m
    if gsd_reference != flight_reference:
        raise ValueError(f"provenance gsd_reference_m {gsd_reference} != flight {flight_reference}")
    model = build(kind, arch, int(provenance["in_channels"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    gsd_min = (float(provenance["gsd_min_m"][0]), float(provenance["gsd_min_m"][1]))
    gsd_max = (float(provenance["gsd_max_m"][0]), float(provenance["gsd_max_m"][1]))
    required = required_gsd_coverage()
    if required is None:
        raise ValueError("required flight GSD coverage could not be computed")
    covered = coverage_ok(gsd_min, gsd_max, required)
    if not covered and not cfg.allow_partial_gsd:
        raise ValueError(
            f"training GSD coverage {gsd_min}..{gsd_max} does not span required "
            f"{required.minimum_m}..{required.maximum_m}; pass allow_partial_gsd "
            "to record the gap"
        )

    torch.manual_seed(0)
    image = torch.randn(1, _TRACE_CHANNELS, TILE_HW[0], TILE_HW[1])
    gsd = torch.zeros(1, 2)
    dynamic_axes: dict[str, dict[int, str]] = {"gsd": {0: "batch"}}
    if cfg.dynamic_spatial:
        dynamic_axes["image"] = {0: "batch", 2: "height", 3: "width"}
        dynamic_axes["logits"] = (
            {0: "batch", 2: "height", 3: "width"} if kind == "segmentor" else {0: "batch"}
        )
    else:
        dynamic_axes["image"] = {0: "batch"}
        dynamic_axes["logits"] = {0: "batch"}

    artifact.parent.mkdir(parents=True, exist_ok=True)
    tmp_artifact = _private_sibling(artifact)
    tmp_sidecar = _private_sibling(sidecar)
    try:
        torch.onnx.export(
            model,
            (image, gsd),
            str(tmp_artifact),
            input_names=["image", "gsd"],
            output_names=["logits"],
            opset_version=cfg.opset,
            dynamo=False,
            dynamic_axes=dynamic_axes,
        )
        _refine_shape_metadata(tmp_artifact)
        image_shape, gsd_shape, output_shape = _verify_graph(tmp_artifact, kind)
        sha256 = hashlib.sha256(tmp_artifact.read_bytes()).hexdigest()
        manifest = ModelManifest(
            version=sha256[:16],
            kind=kind,
            arch=arch,
            sha256=sha256,
            dataset_hash=str(checkpoint["dataset_hash"]),
            band_names=band_names,
            input_shape=image_shape,
            gsd_input_shape=gsd_shape,
            output_shape=output_shape,
            gsd_reference_m=gsd_reference,
            gsd_min_m=gsd_min,
            gsd_max_m=gsd_max,
            conditioning=CONDITIONING_ID,
            gsd_encoding=GSD_ENCODING,
            norm=str(provenance["norm"]),
            coverage_altitude_m=required.altitude_m,
            partial_gsd=not covered,
        )
        write_manifest(tmp_sidecar, manifest)
        _publish(tmp_artifact, artifact)
        try:
            _publish(tmp_sidecar, sidecar)
        except OSError:
            # Sidecar publish failed; remove our own artifact link only while it
            # is still the same file we just linked — never a foreign output.
            if artifact.exists() and os.path.samefile(artifact, tmp_artifact):
                artifact.unlink()
            raise
    finally:
        tmp_artifact.unlink(missing_ok=True)
        tmp_sidecar.unlink(missing_ok=True)
    return artifact


def export(cfg: ExportConfig) -> Result[Path, str]:
    """Export ``cfg.checkpoint_path`` as a validated two-input ONNX artifact.

    Returns:
        Ok(artifact path) on success, else Err. Existing artifact or sidecar
        paths are refused rather than overwritten.
    """
    try:
        return Ok(_export(cfg))
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        return Err(str(exc))
