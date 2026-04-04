# preprocessing/ — PACT Subsystem Context

## Purpose
Transform raw camera frames into normalised model-ready tensors and compute per-frame
quality flags that gate downstream inference.

## Satisfies
- REQ-AIML-PREP-001 — band selection from raw multispectral frames
- REQ-AIML-PREP-002 — radiometric calibration (dark frame + flat field)
- REQ-AIML-PREP-003 — ROI crop and coordinate back-projection
- REQ-AIML-IMAG-002 — quality flag computation gating inference
- REQ-AIML-DATA-003 — per-frame usability classification

## Owns
- `ProcessedFrameMsg` — produced as an in-memory Python object by the preprocessing
  pipeline. NOT serialised to a queue or disk at this stage (see ADR-001 below).

## Consumes
- Fields of `RawFrameMsg`: `raw_bands` (np.ndarray, (C, H, W) float32), `exposure_us`,
  `gain_db`, `gimbal_az_deg`, `gimbal_el_deg`.

## Key Invariants
- Runs inside the inference process as a plain function call — there is no separate
  preprocessing process or thread (see `preprocessing/adr/ADR-001`). This keeps the
  hot-path latency tight by avoiding a serialise→queue→deserialise round trip.
- All functions are pure: they take inputs and return outputs. No global state, no file
  I/O, no side effects except structlog log lines. This makes them trivially testable.
- Functions that can fail return `Result[T, FaultCode]` — they do NOT raise exceptions.
- `apply_calibration()` returns `Err(FaultCode.INFERENCE_NAN)` if output contains NaN
  after dark-frame subtraction and flat-field correction.

## Concurrency
None — the preprocessing module runs synchronously inside the inference
`multiprocessing.Process`. See `preprocessing/adr/ADR-001`.

## Known Gaps / TODOs
- `MOTION_SMEAR` flag in `quality.py` is a placeholder. It is raised based on gimbal
  slew rate but the actual motion estimation is not yet implemented.
- `CLOUD_CONTAMINATED` and `SUNGLINT` thresholds in `quality.py` use hard-coded
  constants with `# TODO: move to config` comments. These need empirical tuning from
  real on-orbit imagery before flight.
- Quality thresholds (`SATURATION_FRACTION_THRESHOLD`, `NIR_RED_RATIO_THRESHOLD`,
  `SUNGLINT_NIR_MEAN_THRESHOLD`) are not yet part of `PactConfig` — they will be
  added when the preprocessing config dataclass is introduced.
