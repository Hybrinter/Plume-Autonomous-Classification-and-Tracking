# tools.ml_models.analysis.visuals.dataset

**Source:** `packages/tools/src/tools/ml_models/analysis/visuals/dataset.py`
**Kind:** module

## Purpose

This module renders captured dataset preview bytes into gallery figures.
Every panel draws the exact captured canonical float32 array: bytes are
checksum-verified against the frozen preview identity, loaded through
`np.load` on in-memory buffers only, and displayed with the supplied
semantic channel mapping. Gallery families that are unavailable or
skipped still produce indexed placeholder figures.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `render_dataset_visuals` | function | Export every planned gallery and family placeholder |

## Inputs and outputs

`render_dataset_visuals(captured: DatasetPreviewCapture, cfg: PlotConfig)
-> Result[RenderedDatasetFigures, str]` returns the gallery bundle files
under `visuals/` plus availability records. The shared record type comes
from `tools.ml_models.analysis.plots.dataset`.

## Behavior

- Each `DatasetPreview` NPZ is verified against its recorded size and
  SHA-256 before decoding; a size or checksum mismatch returns `Err`.
- Three display indices transpose the channel-first array to `H, W, 3`
  in exactly the supplied order; one display index renders grayscale on
  a fixed `0..1` scale with the supplied display label.
- `representative` galleries draw the input panel beside the explicit
  mask panel. An absent mask entry shows `Mask unavailable`; an
  explicit empty mask renders empty and is titled with its recorded
  pixel count. The title carries the tile id and the positive or
  negative label.
- `augmentation` galleries apply every supplied element through
  `apply_dihedral`, label each panel with the element name, and mark
  the figure as verified stored training transforms. No re-selection or
  numerical measurement occurs.
- `same_observation_gsd` galleries draw one input per supplied variant,
  label each with its actual lateral/along GSD, and name the recorded
  observation id; no independence claim is made.
- A `SKIPPED` or `UNAVAILABLE` plan output for a family produces a
  `gallery_unavailable_<family>` placeholder figure with the explicit
  reason, indexed as `visual:gallery_unavailable_<family>`.

Plan outputs pass through unchanged; each rendered gallery adds a
`visual:<identifier>` `AVAILABLE` record. Input tensors are never
resized; only normal viewport scaling applies inside the configured
figure dimensions.

## Errors and faults

Returns `Err` when a preview file is absent, its bytes differ from the
recorded size or checksum, an unknown gallery family is supplied, or
figure export fails.

## Messages

None.

## Configuration

`PlotConfig`; see
[`tools.ml_models.analysis.config`](../config.md).

## Constraints

- Only captured bytes are read; no dataset traversal, source files, or
  model calls occur during rendering.
- No mask is manufactured for a variant that stores none, and no
  normalization or band substitution is applied.

## Related documents

- [`tools.ml_models.analysis.visuals`](../visuals.md)
- [`tools.ml_models.analysis.dataset_previews`](../dataset_previews.md)
- [`tools.ml_models.analysis.visuals.selection`](selection.md)
- [`tools.ml_models.analysis.dataset_render`](../dataset_render.md)
