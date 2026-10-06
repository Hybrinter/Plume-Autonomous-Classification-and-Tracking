"""Figure/export conventions and the render boundary.

``export_figure`` encodes a rendered matplotlib figure into typed
``BundleFile`` bytes per ``PlotConfig`` format without touching user
output; ``render_analysis`` remains unavailable until the plotting phase
lands and still fails closed with an explicit error and no output
directory.

Contains:
  - export_figure: deterministic per-format figure encoding to bundle bytes.
  - render_analysis: the public ``Result`` boundary (unavailable).

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
    """Refuse to render while figure generation is unimplemented.

    Args:
        evidence_dir: Frozen evidence directory.
        cfg: Figure output settings.
        out: Destination figure directory.

    Returns:
        Result[Path, str]: Always Err; no figures are created.
    """
    del evidence_dir, cfg, out
    return Err("figure rendering is unavailable until the plotting phase is implemented")
