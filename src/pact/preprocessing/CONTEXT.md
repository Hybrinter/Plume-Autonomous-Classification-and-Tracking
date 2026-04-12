# preprocessing/ -- Agent Context

## Purpose

Transforms raw camera bands into model-ready tensors and computes per-frame quality flags.
Runs entirely inside the inference process as function calls -- never as a separate process.

## Defining Design Decision

Quality flags are a `frozenset[FrameUsabilityTag]` -- a frame can carry multiple flags
simultaneously (e.g., `SATURATED | CLOUD_CONTAMINATED`). Flagged frames are **not
rejected** -- they are passed through inference and stored with flags set, so ground
operators can curate the dataset using flag metadata. Filtering happens on the ground,
not on-orbit.

## Invariants

- All functions are pure: no global state, no file I/O, no queue access.
- `apply_calibration()` returns `Err(FaultCode.INFERENCE_NAN)` on NaN output -- it does
  not raise. Treat its return as a `Result`.
- `compute_quality_flags()` signature: `(bands: object, exposure_us: float,
  utc_timestamp: str, cfg: PreprocessingConfig) -> frozenset[FrameUsabilityTag]`.
  There is no `gain_db` parameter (removed).

## Gotchas

Band indexing after selection is hardcoded in comments: B2=index 0, B3=1, B4=2, B8=3.
Any change to `input_bands` in config requires reviewing all preprocessing functions that
index into the band array by position.

## Phase II Gaps

`MOTION_SMEAR` flag is a placeholder -- the gimbal-slew-rate heuristic is not yet
implemented. The flag is never raised in Phase I.
