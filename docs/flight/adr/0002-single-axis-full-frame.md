# ADR-FLIGHT-0002: Single-axis envelopes and full-frame inference

**Status:** Accepted
**Date:** 2026-09-03
**Topic:** feature-remove
**Supersedes:** none
**Superseded-by:** none
**Related:** ADR-REPO-0007, ADR-REPO-0008

## Context

The purchased camera and gimbal stack is a single elevation axis behind a 2448 x 2048
mosaic and a 150 mm optic. Flight config still described a dual-axis placeholder, a
square 1024 mosaic, search-mode decimation, a TRACKING ROI crop, and a SCAN azimuth
raster. Those numbers drove pointing IFOV, inference tensor size, and SAFE thermal
compare against a single scalar housekeeping channel that is not a per-component sensor.

Closed-loop pointing, `GimbalRequest`, encoder runaway, and SAFE stow from
ADR-REPO-0008 remain. This record replaces the vehicle numbers, the crop/scale
preprocess path, SCAN, deadband RATE suppression, and thermal compare-on-sample.

## Decision

- Hardware elevation travel is `[-45, +90]` deg signed off-nadir (`+` along-track).
  Science imaging is `[0, +45]`. Stow is `-45`. Home / wait pose is `+45` (science limb).
  Hardware slew is `10 deg/s`. Drivers pin azimuth at `0`. The HAL Protocol stays two-axis.
- Mosaic is `2448` (lateral) x `2048` (along-track). The demosaiced band plane and the
  inference tensor are `1024 x 1224`. `PayloadApp` always passes the full selected band
  plane into `detect()`. Crop and scale are deleted. Pointing uses `ifov_band_deg_per_px`.
- Inference expected latency is `100 ms`. FDIR detect timeout is `500 ms` (`5x`).
  `OnnxDetector` uses `fault.inference_timeout_ms`. Factory ONNX graphs stay at the
  shipped `256 x 256` contract until re-export.
- The arbiter replaces `GimbalState.SCAN` with `REWIND`: TRACKING loss below the science
  limb commands `ABSOLUTE` elevation to `+45`. At the limb, TRACKING holds at rate 0.
  Smear-limited rewind from ISS/TLE is out of scope.
- Deadband RATE suppression is removed. LQR output clamps to hardware slew.
- `[thermal]` records datasheet min/max per component. `ThermalApp.sample()` publishes
  `thermal_sample` and does not emit `THERMAL_OVER_LIMIT`. `SET_THERMAL_LIMIT` still acks
  and stores an override. The fault code and SAFE policy stay for later sensors.

## Consequences

- `compute=real` ONNX load fails until weights are re-exported at `1024 x 1224`. That is
  intended. SIL uses `ScriptedDetector`.
- Azimuth RATE from LQR is ignored by the drivers. A later controller pass can shrink
  LQR to one axis.
- REQ-SAFE-HIGH-002 (thermal over-limit -> SAFE) is a VCRM gap until per-component
  sensors exist. Power over-limit and watchdog remain live SAFE demonstrations.
- ADR-REPO-0008 is not superseded. Its HAL, `GimbalRequest`, and SAFE-stow decisions
  still hold. ROI crop, deadband, and SCAN in that record no longer match the code.

## Alternatives considered

- Keep crop/scale and only retune IFOV — leaves a search-decimation contract the new
  mosaic does not need.
- Fault the current one-channel housekeeping sample against the camera datasheet max —
  would SAFE on a sensor that is not the camera case.
- Fully supersede ADR-REPO-0008 — would drop closed-loop HAL and SAFE stow that remain.
