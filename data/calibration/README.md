# Calibration Artifacts

This directory holds the per-pixel radiometric calibration artifacts for the PACT imaging sensor.
Nothing binary is committed to this repository; the `.npy` files are produced by the sensor
characterization campaign (HIL phase) and are distributed out-of-band.

## Manifest format

Each deployed calibration directory must contain a `manifest.json` with the following structure:

```json
{
  "dark_frame":      {"file": "dark_frame.npy",      "sha256": "<hex>"},
  "flat_field":      {"file": "flat_field.npy",      "sha256": "<hex>"},
  "bad_pixel_mask":  {"file": "bad_pixel_mask.npy",  "sha256": "<hex>"}
}
```

- `file`: path relative to this directory.
- `sha256`: lowercase hex SHA-256 digest of the raw bytes of the `.npy` file (including the
  numpy format header).

`flight.payload.calibration_io.load_calibration` reads this manifest, verifies each digest, and
checks that every array shape matches the `SensorConfig` geometry before accepting the artifacts.
Any mismatch returns `Err(CALIBRATION_INVALID)`, which the composition root treats as an
unrecoverable startup failure.

## Artifact descriptions

| Artifact        | dtype   | shape    | Description                                              |
|-----------------|---------|----------|----------------------------------------------------------|
| `dark_frame`    | float32 | (H, W)   | Per-pixel dark signal in DN, measured at operating temp. |
| `flat_field`    | float32 | (H, W)   | Normalized per-pixel response map; values near 1.0.      |
| `bad_pixel_mask`| bool    | (H, W)   | True where a pixel is unusable; repaired by interpolation.|

## Identity calibration (SIL/dev only)

When `SensorConfig.calibration_dir` is empty (`""`), the composition root calls
`flight.payload.calibration_io.build_identity_calibration` instead of loading artifacts from
disk. This produces zero dark signal, unit flat field, and no bad pixels. It is intended
exclusively for simulation and development -- it MUST NOT be used in flight.
