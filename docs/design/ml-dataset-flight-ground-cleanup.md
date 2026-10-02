# ML dataset flight and ground cleanup

**Status:** Flight processing and stacked ground workflow integrated; stack validation complete.
**Date:** 2026-10-02

## Objective and evidence

Align the ML workflow with flight dataset collection and ground dataset
preparation, while reusing the previous agents' useful implementations.
The user identified an omitted flight package as the cause of misplaced work.
This brief records the initial review; it does not turn the prior agent's
proposal into approved requirements.

Inputs reviewed:

- The user-supplied `Dataset Generation.svg`.
- The user-supplied prior plan dated 2026-10-01.
- The local checkout at `70404c8`, containing the merged ML data foundation
  and architecture builders.
- Remote source and patches for open pull requests 100 through 104. These
  contain newer dataset, footprint, conditioning, training, and export work
  than the returned pull-request descriptions describe. The branch heads were
  fetched before stack integration; a branch name is not a pinned snapshot.

The diagram includes two connected paths:

1. Flight processes an image, splits 64 tiles, classifies them, segments the
   classifier-positive tiles, and computes plume geometry for control.
   It retains classifier-positive examples and a balancing selection of
   classifier-negative examples. It also retains segmentor-positive examples
   and a balancing selection of segmentor-negative examples. Products pass
   through flight storage and downlink.
2. Ground receives a raw flight dataset. A separate Zenodo path selects bands
   and resamples to several GSDs. Both paths then split train/validation/test,
   augment training examples, write a finished dataset, and run training.

The user confirmed that flight owns preprocessing, geometry/GSD, tiling,
inference, storage selection, and storage. Ground work begins after positive
and negative examples are compiled. The current implementation pass covers
processing and the new flight model interface. Example collection, balancing,
training-example storage, and downlink are future work.

The future collection policy will retain negatives in a configurable ratio
to positives, defaulting to 1:1 independently for the classifier and segmentor.
That ratio is not a train/validation/test split. No ratio setting or sampling
policy is added during this pass. Label authority and persisted pixel format
remain decisions for the future collection/ground-ingestion work.

## Confirmed ownership boundary

Flight owns the reusable functions required for live processing. Tools consumes
those functions for ground workflows. Future collection ownership is recorded
here without implementing collection during this pass.

| Responsibility | Owner | Ground reuse |
| --- | --- | --- |
| Camera calibration, quality, normalization | `flight.payload.preprocess` | Call flight functions for the same pixel domain |
| Ray geometry and measured per-tile GSD | `flight.payload.gimbal` | Call pure footprint functions for source adaptation |
| Tile layout, slicing, mask stitching | Flight payload pure core | Import the flight implementation |
| Classification gate, selected segmentation, blob extraction | `flight.payload.inference` | Evaluate through the shared runtime contract |
| Onboard example selection and collection metadata | Flight payload pure policy plus app shell | Decode products into raw ground records |
| Durable storage and downlink | Existing flight HAL/core/ISS paths | Ingest the delivered products |
| Raw-source adapters, group splitting, offline augmentation, finished datasets | `tools.ml_models.dataset` | Ground-only workflow |
| Torch model builders, training, checkpoint evaluation | `tools.ml_models.arch` and `train` | Ground-only workflow |
| ONNX export and promotion eligibility | `tools.ml_models.export` | Validate against flight-owned contracts |
| Model studies and report generation | `tools.ml_models.studies` and `analysis` | Keep separate from SIL recording in `tools.analysis` |

Flight must stay independent of tools and Torch. Reuse points in flight must
be pure NumPy/math functions or typed backend interfaces. Tools may import
those APIs. Physical frame geometry and deployment behavior must not be
independently specified by a tools dataset adapter.

## Reuse inventory

Retain or adapt the following work after checking its exact branch snapshot:

- Group-disjoint split logic and dataset provenance/validation from the merged
  `tools.ml_models.data` foundation.
- The finished dataset manifest, per-task/per-shape shards, atomic build path,
  loader, flight directory reader, and synthetic source from the newer PRs.
- The canonical Zenodo archive reader, annotations, band adaptation, and
  resampling modules. Unannotated masks must remain unknown.
- The existing flight ray/ellipsoid geometry and the newer flight footprint
  helper, including model-GSD encoding.
- The architecture registry, checkpoint behavior, losses, metrics, training
  loop, FiLM extension, and two-input export work where their contracts match
  the settled design.
- Model study calculations and reports, moved to their agreed ground home.

Do not preserve the following assumptions merely to retain old structure:

- Flight frame/grid constants and slicing owned by
  `tools.ml_models.dataset.geometry`.
- A tools-only implementation of the live classifier/segmentor gate.
- Pasting small Zenodo examples onto negative mosaics to manufacture flight
  frames as the ordinary dataset path.
- Independent data formats, split recipes, polygon parsers, and normalization
  recipes in training, source adapters, and studies.
- A tools export contract that flight's loader and deployer cannot consume.

## Integration gaps to resolve

### Pixel-domain boundary

Flight calibration produces floating-point corrected DN, then normalization
clips/scales to a unit tensor. The proposed raw flight directory instead
specifies uint16 DN. Decide what pixels are actually persisted: sensor DN,
corrected DN, or quantized unit pixels. Persist domain, scale, band order,
calibration identity, and processing identity explicitly. Ground preparation
must neither apply calibration twice nor normalize a unit image as ADC DN.

### Selection versus ground truth

The diagram selects examples using current model outputs. A stored prediction
is provenance and a selection reason; whether it is also a training label is
unresolved. Preserve classifier logits, segmentation outputs where required,
model version, thresholds, and reviewed-label status independently. A
classifier-negative tile has no observed segmentor mask if segmentation was
skipped. It must not silently receive an empty ground-truth mask.

### Selection versus complete-frame evaluation

The diagram deliberately discards surplus negatives. PR104's evaluator only
scores complete 64-tile frames. Ordinary collected training examples may not
provide complete frames. Define a separate complete-frame validation capture
path or specify tile-level evaluation and report incomplete frames honestly.
Missing tiles cannot be replaced with fabricated negatives.

### Storage and link budgets

The diagram states 1 GB/day. Current configuration contains both
`max_daily_downlink_bytes = 1073741824` and
`downlink_max_bytes_per_pass = 1048576`. The current downlink drain reads only
the latter and resets its local accounting each call. The daily setting is
not used by that path. These settings are not evidence of enforced daily or
contact-wide limits.

The target must distinguish capture/storage allocation, retained class
balance, total delivered bytes, actual contact budget, and metadata/mask
overhead. Retaining all positives is contingent on a finite storage policy.
Use the existing storage-reference path; large pixel arrays stay off the bus.
Budget enforcement changes are outside this implementation pass.

### Geometry and GSD

Use measured shutter-time orbit and encoder data where available. The source
of nominal fallback values and their quality tags must be explicit. The newer
footprint helper states that positive elevation yields smaller footprints
toward the bottom of the image; the prior plan predicted the opposite
monotonic trend. Tests must follow the existing coordinate transformations.

GSD is ordered lateral/W then along-track/H. Augmentation must preserve that
relationship: shape preservation alone does not prove that a transform is
legal for anisotropic GSD. Axis-swapping transforms must either update the
GSD pair consistently or be excluded under the agreed orientation convention.
Zenodo resampling is a geometric/radiometric proxy and does not prove flight
accuracy or latency. Keep deployment eligibility separate from measured
performance evidence.

### Coordinated deployment

At the initial review, the local flight detector and ONNX loader used the
single-input full-frame contract. The newer proposed exporter emits a
dynamic-batch image/GSD tile pair. A usable cutover must update flight loading,
verification, model-pair
activation/rollback, app inputs, detector backends, and SIL substitutes in a
coherent sequence. An intermediate exporter must not promote artifacts that
the deployed flight consumer cannot load.

## Migration and delegation sequence

1. Reuse the prior footprint implementation and tile algorithms, adapting
   them to flight's pure Result APIs and configuration. Flight retains full
   sensor dimensions while the model receives smaller tiles.
2. Use the new standard: dynamic-batch `image (B,3,193,258)` and `gsd (B,2)`
   inputs for the default 8x8 grid. The GSD pair is lateral/W then along/H;
   model values are `ln(gsd_metres / gsd_reference_m)` as float32.
3. Classify every tile; segment only classifier-positive tiles; scatter
   probabilities into zero masks for negative tiles; stitch the full frame
   before existing blob extraction and control.
4. Prepare shutter-time GSD before inference. Reuse orbit/angle observations
   for control. Tag any nominal fallback as `GSD_NOMINAL`.
5. Coordinate strict named-input loading and pair deployment/rollback with
   the same dynamic-batch tile contract. Reject old single-input artifacts;
   preserve their real metadata until compatible trained models are exported.
6. Exercise the tiled flow in deterministic SIL and in actual two-input
   ONNX tests. Review agent changes and run repository checks.

Four GPT-6-luna execution tasks have disjoint ownership: geometry/tiling/config,
inference backends and gate, session/deployment contract, and app/message/SIL
integration. The parent reviews integration and maintains cross-cutting docs.
The user subsequently requested updates to the existing PR 100–104 stack;
integration preserves branch histories and uses fast-forward publication.

The existing pull-request stack now carries these APIs through finished ground
datasets, conditioning, training, export, and evaluation. Collection, storage,
balancing, and downlink policies remain future work. Existing mask science
products continue through the current storage path during this pass.

### Pull-request allocation

| Pull request | Resulting responsibility |
| --- | --- |
| 100 | Flight processing/model interface plus finished dataset core; tools geometry and GSD encoding delegate to flight |
| 101 | Ground Zenodo decoding, annotation adaptation, spectral mapping, and resampling; flight supplies reference geometry |
| 102 | Torch FiLM conditioning and training; export-size metadata derives from flight-backed geometry |
| 103 | Ground two-input export, acceptance, and pairing; actual flight loader and deployment validate generated fixed-tile artifacts |
| 104 | Ground analysis and legacy cutover; full-frame scoring calls flight's classifier/segmentor gate and stitching |

Each layer retains its existing branch history and incorporates its updated
parent. GitHub's native stack preserves PR 100's base at the already-merged
PR 99 and rejects direct retargeting; that base tree matches current main,
which is incorporated into the updated head. Ground
augmentation, group splitting, finished dataset shards, training, and reports
stay in tools because they operate after examples are compiled. Ground raw
directory writers are import/fixture helpers, not onboard collection services.

The default export has dynamic batch and concrete flight tile dimensions.
Dynamic spatial graphs remain research artifacts and fail flight promotion
and pairing. ONNX shape inference resolves segmentor Resize dimensions before
validation; actual Runtime parity and deployment tests check the resulting
graph metadata.

Every execution task should identify its exact files, input/output contract,
reuse source, forbidden dependencies, acceptance checks, and out-of-scope
decisions. Agents share a workspace; assign disjoint files or serialize work
on cross-package contracts.

## Acceptance evidence

- Slice/stitch round trips and measured geometry tests with the actual axis
  convention; malformed data follows flight's Result contract.
- Shared flight and offline gate behavior, with segmentation called only for
  classifier-positive tiles and finite outputs checked.
- Two-input ONNX parity and dynamic-batch checks; generated classifier and
  segmentor artifacts pass acceptance, pairing, flight parsing, and activation.
  The legacy ground exporter is removed by the stack's cutover layer.
- SIL pointing and model-upload scenarios pass after wiring.
- Relevant static, import, documentation, requirement, flight-image, and
  subsystem test gates pass. Hardware timing and flight-domain accuracy
  remain measured validation tasks, not conclusions from synthetic tests.

Flight-foundation validation before stack integration, with optional ONNX
dependencies installed:
`pytest -m 'not e2e' -n 2` passed 1,168 tests and skipped 12. Ruff lint/format,
strict mypy, all 18 import contracts, strict documentation/ADR checks, VCRM,
and the isolated lean flight-image check passed. Verification includes actual
two-input ONNX execution, tiled geometry and gating, SIL pointing, model upload,
activation/rollback, and analysis scenarios. The broader run also exposed stale
ground export fixtures and a finalizer channel-count assumption; those were
corrected while preserving three- and four-band finalization support.

Stack validation includes 1,105 passing PR-suite tests on PR 100 and 1,134 on
PR 101; 35 architecture/training tests on PR 102; and all 122 export tests on
PR 103. The integrated run exercised 1,136 tests, with 1,128 passing and four
skipped. Its four failures exposed stale dynamic-spatial contract expectations
and handcrafted ONNX fixtures using an unsupported IR version. Contract
checks now distinguish fixed flight tiles from dynamic research graphs; the
precision fixtures use supported IR 10 and declare their actual input shape.
Both real SDK precision conversions pass. After these fixture corrections,
the final PR suite (`pytest -m 'not slow and not e2e' -n 2`) passed 1,103 tests
and skipped four. All static gates and the isolated flight image check pass
on the integrated stack.

## Future decisions outside this pass

1. Label authority, persisted pixel domain, calibration provenance, and
   ground review before selected predictions become training targets.
2. Collection time horizon, configurable negative/positive ratio, capacity
   behavior, and handling of incomplete frames for evaluation.
3. Actual daily/contact transport budgets and retention/eviction policy.
4. The ground migration's final augmentation/GSD variants and source population
   details. These do not block the confirmed flight model interface.

Future collection and source-population work will need these decisions; this
pass implements the processing boundary and reuses it throughout the stack
without selecting the deferred collection policies.
