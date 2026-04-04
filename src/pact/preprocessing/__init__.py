"""
pact.preprocessing — Raw-frame-to-tensor pipeline for PACT inference.

Satisfies: REQ-AIML-PREP-001, REQ-AIML-PREP-002, REQ-AIML-PREP-003,
           REQ-AIML-IMAG-002, REQ-AIML-DATA-003

Runs synchronously inside the inference multiprocessing.Process (no separate process).
All public functions are pure (no side effects except logging).

Submodules:
    band_select    — BAND_INDICES constant and select_bands()
    radiometric    — RadiometricCalibration dataclass and apply_calibration()
    quality        — compute_quality_flags() for per-frame usability classification
    crop           — crop_to_roi() and backproject_pixel() for ROI windowing
"""
