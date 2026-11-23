"""Figure/export conventions and the render boundary.

``export_figure`` encodes a rendered matplotlib figure into typed
``BundleFile`` bytes per ``PlotConfig`` format without touching user
output; ``render_analysis`` re-renders a verified published bundle from its
frozen recipes into a fresh exclusive output.

Contains:
  - export_figure: deterministic per-format figure encoding to bundle bytes.
  - render_analysis: the public ``Result`` render-only boundary.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from flight.libs.types import Err, Ok, Result

from tools.ml_models.analysis.artifacts import BundleFile
from tools.ml_models.analysis.config import PlotConfig

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from tools.ml_models.analysis.contracts import ArtifactKind

_SVG_HASH_SALT = "pact-evidence"


def export_figure(
    figure: Figure,
    identifier: str,
    cfg: PlotConfig,
    *,
    kind: ArtifactKind = "FIGURE",
    population: str | None = None,
) -> Result[tuple[BundleFile, ...], str]:
    """Encode one rendered figure into bundle bytes for every configured format.

    Bytes are produced through in-memory buffers only; no output directory is
    reserved or written. The supplied figure is always closed, even when a
    ``savefig`` call fails. A fixed SVG hash salt and omitted date metadata
    keep identical content byte-stable where the backend permits.
    """
    del population
    import matplotlib

    try:
        if kind not in ("FIGURE", "VISUAL"):
            return Err(f"cannot export figure {identifier}: unknown artifact kind {kind!r}")
        matplotlib.use("Agg")
        figure.set_size_inches(float(cfg.width_inches), float(cfg.height_inches))
        prefix = "figures" if kind == "FIGURE" else "visuals"
        files: list[BundleFile] = []
        with matplotlib.rc_context({"svg.hashsalt": _SVG_HASH_SALT}):
            for fmt in cfg.formats:
                buffer = BytesIO()
                metadata: dict[str, str | None] = {}
                if fmt == "svg":
                    metadata = {"Date": None}
                elif fmt == "pdf":
                    metadata = {"CreationDate": None, "ModDate": None}
                figure.savefig(buffer, format=fmt, dpi=cfg.dpi, metadata=metadata)
                files.append(
                    BundleFile(path=f"{prefix}/{identifier}.{fmt}", data=buffer.getvalue())
                )
        return Ok(tuple(files))
    except (OSError, ValueError, RuntimeError) as exc:
        return Err(f"cannot export figure {identifier}: {exc}")
    finally:
        figure.clf()
        import matplotlib.pyplot as plt

        plt.close(figure)


def render_analysis(evidence_dir: Path, cfg: PlotConfig, out: Path) -> Result[Path, str]:
    """Re-render a verified evidence bundle without re-measuring anything.

    The bundle is checksum-verified first; only referenced files, versioned
    recipe documents and the frozen scientific document are read. Model and
    dataset bundles dispatch to their own renderers; a missing or corrupt
    recipe returns an actionable "fresh analyze required" error and no output
    is created. The destination is exclusive and never inside the bundle.

    Args:
        evidence_dir: Frozen evidence directory.
        cfg: Figure output settings.
        out: Destination figure directory.

    Returns:
        Result[Path, str]: The published bundle path, or an explicit error.
    """
    from tools.ml_models.analysis.artifacts import verify_bundle
    from tools.ml_models.analysis.summaries import DatasetSummary, ModelTrainingSummary

    root = Path(evidence_dir)
    verified = verify_bundle(root)
    if isinstance(verified, Err):
        return verified
    summary = verified.value
    if isinstance(summary, ModelTrainingSummary):
        from tools.ml_models.analysis.model_render import render_model_bundle

        return render_model_bundle(root, summary, cfg, Path(out))
    if isinstance(summary, DatasetSummary):
        from tools.ml_models.analysis.dataset_render import render_dataset_bundle

        return render_dataset_bundle(root, summary, cfg, Path(out))
    return Err(f"bundle {root} has an unsupported summary kind")
