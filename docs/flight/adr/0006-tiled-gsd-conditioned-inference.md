# ADR-FLIGHT-0006: Flight-owned tiled GSD-conditioned inference

**Status:** Accepted
**Date:** 2026-10-01
**Topic:** interface
**Supersedes:** ADR-FLIGHT-0003 (model graph contract only)
**Superseded-by:** none
**Related:** ADR-FLIGHT-0001, ADR-FLIGHT-0002, ADR-TOOLS-0013

## Context

The camera provides a three-band 1544 by 2064 frame. The newer ground model
workflow exports tile-sized classifier and segmentor graphs conditioned on
ground sample distance. The flight detector and deployer still describe a
single-input full-frame graph, while ground tools contain their own tiling
and classifier/segmentor orchestration. Those implementations disagree on
the contract consumed by the payload.

The user confirmed that flight owns preprocessing, geometry/GSD, tiling,
inference, example selection, and storage. Ground dataset preparation starts
after examples are compiled. This pass implements processing and the model
interface; collection, configurable class balancing, training-example storage,
and downlink are deferred.

## Decision

- Flight owns pure footprint and tile slicing/stitching functions. Ground
  tools may consume these APIs; flight imports neither tools nor Torch.
- The frame dimensions stay full sensor size. The default 8 by 8 grid yields
  193 by 258 pixel tiles in row-major order. The grid is configurable and must
  divide the frame dimensions.
- Each ONNX graph has exactly two float32 inputs: `image (B,C,h,w)` and
  `gsd (B,2)`. Batch is dynamic; deployed spatial dimensions match the tile.
  The classifier returns `(B,1)` logits and the segmentor returns
  `(B,1,h,w)` logits. Flight applies sigmoid to segmentor logits.
- GSD is ordered lateral/W then along-track/H and encoded as
  `ln(gsd_metres / gsd_reference_m)`. The configured reference defaults to
  15.87 metres. Orbit and encoder observations are taken at shutter time
  before inference. Nominal fallback geometry is marked `GSD_NOMINAL`.
- The classifier runs over every tile. Only classifier-positive tiles reach
  the segmentor. Negative tiles receive zero masks. Flight stitches the mask
  before the existing full-frame blob and controller logic.
- Loader and model-pair deployment validate the same contract. Pair metadata
  must match the configured grid, frame/tile sizes, bands, and GSD reference.
  Incompatible activation preserves the active pair through the existing
  modeled rollback path.
- Existing factory graphs retain their actual metadata and are rejected by
  the new consumer. Compatible trained artifacts must be exported before
  real inference is available. Test graphs are not factory model replacements.

## Consequences

- Shared pure flight APIs replace tools-owned runtime behavior as the ground
  workflow is migrated. Ground model builders and training remain in tools.
- Tiled gating and GSD data flow can be tested with deterministic SIL and
  actual ONNX graphs without collecting training examples.
- The previous expected latency and FDIR timeout remain configuration values.
  The full-frame performance estimate does not validate tiled inference;
  hardware measurement is needed before claiming a tiled timing budget.
- The future collection policy defaults to one retained negative per positive
  independently for classifier and segmentor examples, with a configurable
  ratio. No sampling or retention policy is implemented by this decision.

## Alternatives considered

- Keep flight runtime behavior in tools: creates incompatible duplicate
  implementations and would require flight to depend on ground tooling.
- Change exporter metadata without changing the flight consumer: produces
  artifacts that cannot be loaded or activated.
- Relabel old factory graphs as GSD-conditioned: the graph inputs and trained
  behavior do not implement that interface.
