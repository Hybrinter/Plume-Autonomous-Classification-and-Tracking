"""Two-input ONNX session opening and contract validation.

The hash is verified before the SDK loads the artifact. After load, the
session must expose exactly ``image`` and ``gsd`` float inputs and one float
output, with a shared dynamic batch dimension and declared shapes equal to
the manifest. There is no one-input fallback.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from flight.libs.types import Err, Ok, Result
from flight.payload.inference.verify import compute_sha256

from tools.ml_models.export.contract import verify_conditioned_shapes
from tools.ml_models.export.manifest import ModelManifest

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True)
class Node:
    """One declared graph input or output.

    Attributes:
        name: Graph value name.
        shape: Declared shape; symbolic or unknown dims appear as str/None.
        type: Element type string such as ``tensor(float)``.
    """

    name: str
    shape: tuple[int | str | None, ...]
    type: str


class Session(Protocol):
    """Minimal inference-session surface used by acceptance and pairing."""

    def get_inputs(self) -> Sequence[Node]:
        """Return the declared graph inputs."""
        ...

    def get_outputs(self) -> Sequence[Node]:
        """Return the declared graph outputs."""
        ...

    def run(
        self,
        output_names: list[str] | None,
        feeds: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        """Run inference over the given feeds and return output arrays."""
        ...


def _normalize_shape(shape: tuple[int | str | None, ...]) -> tuple[int | None, ...]:
    """Map symbolic dimension names to None, preserving fixed ints."""
    return tuple(dim if isinstance(dim, int) else None for dim in shape)


class RuntimeNode(Protocol):
    """Structural typing for an onnxruntime ``NodeArg``."""

    name: str
    shape: list[int | str | None]
    type: str


class RuntimeSession(Protocol):
    """Structural typing for an onnxruntime ``InferenceSession``."""

    def get_inputs(self) -> list[RuntimeNode]:
        """Return the declared graph inputs."""
        ...

    def get_outputs(self) -> list[RuntimeNode]:
        """Return the declared graph outputs."""
        ...

    def run(
        self,
        output_names: list[str] | None,
        feeds: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        """Run inference over the given feeds and return output arrays."""
        ...


class _OrtSession:
    """Adapter from an onnxruntime InferenceSession to the Session protocol."""

    def __init__(self, inner: RuntimeSession) -> None:
        self._inner = inner

    def get_inputs(self) -> list[Node]:
        return [
            Node(str(node.name), tuple(node.shape), str(node.type))
            for node in self._inner.get_inputs()
        ]

    def get_outputs(self) -> list[Node]:
        return [
            Node(str(node.name), tuple(node.shape), str(node.type))
            for node in self._inner.get_outputs()
        ]

    def run(
        self,
        output_names: list[str] | None,
        feeds: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        return self._inner.run(output_names, feeds)


def _ort_loader(path: str) -> Session:
    """Construct an onnxruntime session on the CPU provider (lazy import)."""
    import onnxruntime

    inner = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _OrtSession(cast(RuntimeSession, inner))


def open_session(
    artifact: str | Path,
    manifest: ModelManifest,
    *,
    loader: Callable[[str], Session] | None = None,
) -> Result[Session, str]:
    """Verify ``artifact`` against ``manifest`` and open a validated session.

    The artifact hash is checked before the runtime loads the file. Input
    names must be exactly ``{"image", "gsd"}``, exactly one output is allowed,
    every declared I/O must be ``tensor(float)``, batch dims must all be
    dynamic with the same named symbol (or all ``None``), and normalized
    declared shapes must equal the manifest shapes and pass the flight shape
    contract.

    Args:
        artifact: Path to the ONNX artifact.
        manifest: Validated sidecar for the artifact.
        loader: Optional session constructor (testing hook); defaults to
            onnxruntime with the CPU execution provider.

    Returns:
        Ok(Session) when every check passes, else Err with a reason.
    """
    artifact_path = str(artifact)
    try:
        digest = compute_sha256(artifact_path)
    except OSError as exc:
        return Err(f"artifact unreadable: {exc}")
    if digest.lower() != manifest.sha256.lower():
        return Err("artifact sha256 does not match the manifest")
    try:
        session = (loader or _ort_loader)(artifact_path)
    except Exception as exc:  # noqa: BLE001 - SDK errors surface as Err
        return Err(f"onnxruntime failed to load {artifact_path}: {exc}")
    try:
        inputs = list(session.get_inputs())
        outputs = list(session.get_outputs())
    except Exception as exc:  # noqa: BLE001
        return Err(f"session metadata unreadable: {exc}")
    if len(inputs) != 2 or {node.name for node in inputs} != {"image", "gsd"}:
        return Err("graph inputs must be exactly {'image', 'gsd'}")
    if len(outputs) != 1:
        return Err("graph must declare exactly one output")
    for node in (*inputs, *outputs):
        if node.type != "tensor(float)":
            return Err(f"{node.name} must be tensor(float); got {node.type}")
        if not node.shape:
            return Err(f"{node.name} declares an empty shape")
    by_name = {node.name: node for node in inputs}
    image, gsd, logits = by_name["image"], by_name["gsd"], outputs[0]
    batch_dims = [node.shape[0] for node in (image, gsd, logits)]
    if all(isinstance(dim, str) for dim in batch_dims):
        if len(set(batch_dims)) != 1:
            return Err("batch dims must share one named symbol")
    elif not all(dim is None for dim in batch_dims):
        return Err("batch dims must be dynamic on every I/O tensor")
    image_shape = _normalize_shape(image.shape)
    gsd_shape = _normalize_shape(gsd.shape)
    output_shape = _normalize_shape(logits.shape)
    if (
        image_shape != manifest.input_shape
        or gsd_shape != manifest.gsd_input_shape
        or output_shape != manifest.output_shape
    ):
        return Err("declared shapes do not match the manifest")
    contract = verify_conditioned_shapes(
        image_shape,
        gsd_shape,
        output_shape,
        len(manifest.band_names),
        manifest.tile_hw,
        manifest.kind,
    )
    if isinstance(contract, Err):
        return Err(f"declared shapes violate the flight contract ({contract.error.value})")
    return Ok(session)
