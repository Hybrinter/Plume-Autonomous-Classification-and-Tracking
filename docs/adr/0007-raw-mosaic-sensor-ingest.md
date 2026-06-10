# ADR 0007: Raw-mosaic sensor ingest contract

**Status:** Accepted (2026-06-09)

## Context

The 2026-06-06 baseline (docs/superpowers/baseline/2026-06-06-pact-flight-parity-baseline.md,
Section 4.4) identified the **sensor-domain mismatch** as the deepest cross-cutting gap: the HAL
Protocol and the entire preprocessing stack assumed already-separated spectral bands delivered as
a `(C, H, W)` `RawFrameMsg`, but the physical device is a **monochrome camera behind a 2x2
mosaic filter** (FLIR Blackfly S + custom BLUE/GREEN/RED/NIR filter, spec Section 2). There was
no demosaic anywhere in the codebase (zero hits for demosaic/debayer/bayer/mosaic in the pre-plan
tree), no dark/flat correction on the raw plane, identity calibration was a hardcoded placeholder,
and motion-smear quality was an exposure-only placeholder with no slew-rate input.

The consequence was that the SIL never exercised the ingest path -- `ScriptedDetector` operated on
zeroed band stacks passed through an effectively no-op pipeline. Any domain drift between the real
sensor and the model input would be invisible until HIL.

Two design questions had to be resolved before implementation:

1. **Where does demosaic live -- driver or preprocess?** The driver option (returning
   already-separated bands) would make calibration ambiguous: dark/flat correction physically
   belongs on the raw mosaic plane (it corrects per-pixel sensor response, not per-band reflectance),
   and putting calibration inside the driver violates the acquire-only driver contract. Preprocess
   is the correct home: pure, testable, replayable, and exercised by the SIL through the full
   ingest path.

2. **What band vocabulary to use?** The legacy `B2/B3/B4/B8` names are Sentinel-2 band IDs that
   carry no physical meaning at the call site. The mosaic filter passbands approximate those
   Sentinel-2 bands (490/560/665/842 nm) so the training dataset remains a valid domain, but the
   PACT sensor is its own device. Renaming to `BLUE/GREEN/RED/NIR` with Sentinel-2 correspondence
   documented at the enum definition decouples the sensor vocabulary from the training-data origin.

## Decision

**Raw-mosaic HAL contract:** `ImagingSensor.acquire_frame()` returns a `MosaicFrame` -- a raw
`(H, W)` uint16 2x2-CFA mosaic plane plus `timestamp_utc`, `frame_id`, `exposure_us`, and
`gain_db`. Drivers acquire only; no demosaic, calibration, or normalization inside any driver.
`RawFrameMsg` (the band-stack bus message) is removed; frames never touch the bus (co-location
invariant preserved).

**Demosaic and calibration in preprocess (pure functions):** the ingest pipeline runs entirely as
pure function calls inside `PayloadApp.process_frame()`, in order:

```
calibrate_mosaic  (bad-pixel repair -> dark/flat on raw mosaic plane)
  -> separate_bands  (2x2 CFA -> 4 registered half-resolution band planes)
  -> normalize_dn    ([0, 1] by ADC full scale; clips calibration undershoot/overshoot)
  -> select_bands    (reorder into InferenceConfig.input_bands order)
  -> compute_quality_flags  (saturation, physical smear model, illumination)
```

This sequence respects the physics: dark/flat runs on the raw plane where the sensor response
lives; demosaic produces co-registered half-resolution planes; normalization defines the model
input domain.

**BLUE/GREEN/RED/NIR band vocabulary:** the `Band` enum replaces `B2/B3/B4/B8` with
`BLUE/GREEN/RED/NIR`; Sentinel-2 correspondence (B2/B3/B4/B8 at 490/560/665/842 nm) is
documented at the definition site. Mosaic cell layout `("BLUE", "GREEN", "RED", "NIR")` in row-
major cell order is part of `SensorConfig`.

**Checksummed calibration artifacts:** dark frame, flat field, and bad-pixel mask are `.npy`
files under `data/calibration/`, SHA-256-verified at startup against `manifest.json`. Any
integrity or shape mismatch returns `Err(CALIBRATION_INVALID)`, which the composition root treats
as an unrecoverable startup failure. Identity calibration (zero dark / unit flat / no bad pixels)
is the SIL-only fallback (selected by `SensorConfig.calibration_dir == ""`).

**Physical smear quality gate:** `MOTION_SMEAR` is now computed as
`smear_px = slew_rate_deg_per_s * (exposure_us * 1e-6) / ifov_deg_per_px`. The slew rate is
derived from consecutive `GimbalActuator.read_position()` calls in the payload app's `run()` loop
(0.0 on the first frame or when a read fails -- the gate degrades gracefully). `ifov_deg_per_px`
is a `SensorConfig` optics constant (0.04 deg/px, the former `PIXEL_TO_DEG` value).

**Fake-PySpin test pattern:** `RealSensor` is tested in CI via a fake `PySpin` module injected by
`monkeypatch.setitem(sys.modules, "PySpin", ...)`, so the lazy-import contract and full driver
behavior are verified without the physical SDK.

## Consequences

- **One ingest path, exercised end-to-end.** The SIL now runs real signal (radiometrically-
  plausible mosaic frames from `sim.scene.plume`) through calibrate -> demosaic -> normalize ->
  select -> quality -> detector, so domain drift between simulation and flight cannot be invisible.

- **Band planes are half-resolution.** The 512x512 mosaic yields 256x256 band planes matching the
  existing model input size. This is inherent to the 2x2 CFA geometry and is not a lossy step.

- **Model input domain is defined by `normalize_dn`.** `clip(dn / (2**bit_depth - 1), 0, 1)` is
  the exact normalization the model repo must replicate at training time to avoid silent domain
  drift (spec Section 4).

- **Calibration is a startup gate.** A bad `calibration_dir` raises `SystemExit` from `main()`
  before the scheduler starts. Flight should not proceed with an uncalibrated sensor; SIL avoids
  the gate by setting `calibration_dir = ""`.

- **`RawFrameMsg` and `MessageType.RAW_FRAME` are removed.** Downstream consumers that depended
  on them must migrate to the processed output (`ProcessedFrameMsg`) or subscribe to a
  bus-published summary. Large tensors never go on the bus.

- **`SensorConfig` added to `PactConfig`.** Mosaic geometry (`width_px`, `height_px`,
  `bit_depth`, `mosaic_layout`), optics (`ifov_deg_per_px`), startup tuning
  (`default_exposure_us`, `default_gain_db`), and `calibration_dir` are now typed config rather
  than hardcoded constants scattered across `control.py` and `app.py`.
