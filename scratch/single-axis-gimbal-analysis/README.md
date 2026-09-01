# TEMPORARY ANALYSIS — single-axis gimbal tracking time

This folder is a working study, not flight software. It is not on the CI path
and it is not a package. Re-run after editing assumptions in `analyze.py`:

```text
uv run python scratch/single-axis-gimbal-analysis/analyze.py
```

Generated numbers and figures:

- [`RESULTS.md`](RESULTS.md) — questions, assumptions, tables
- [`outputs/`](outputs/) — plots and `tracking_time_vs_radius.csv`

## What this answers

Given the BFS-U3-50S5 (IMX264, 3.45 µm, 2448×2048) and a 150 mm athermal C-mount
lens, at ISS altitude, how long can we track a ground-fixed industrial-plume
origin through the operational elevation window, and how much of that time a
single elevation axis (azimuth parked) loses relative to a two-axis gimbal
when the tracked centre of geometry stays in a ground disk of radius R.

The sensor long axis is placed **lateral** (cross-track) so the 3.2 deg side,
not the 2.7 deg side, is the one without a motor.

## Headline (defaults: 400 km, one-sided 90→30 deg, origin on the slice)

| Quantity | Value |
| --- | --- |
| Sensor FOV (2448 lateral × 2048 along) | 3.23 deg × 2.70 deg |
| Nadir GSD (3.45 µm mosaic pixel) | 9.2 m |
| Along-track track time (elevation axis) | **108 s** |
| Staring at nadir, no gimbal | 2.7 s |
| Peak elevation rate | 1.03 deg/s (under the 2 deg/s cap) |
| Nadir lateral half-swath | ±11.3 km |
| Time lost dropping azimuth, R ≤ 10 km | **0 s** |
| Time lost at R = 15 / 20 / 30 km | 47 / 79 / 108 s |

Full tables, assumptions, and figures: [`RESULTS.md`](RESULTS.md).

## Questions that still change the number

Please answer these; the script is built so each one is a constant at the top
of `analyze.py`.

1. **Window sidedness.** Is `[+90, +30]` deg elevation one side of nadir only,
   or can a single elevation axis travel through nadir to 30 deg on the other
   side (~2× along-track time, still one motor)?
2. **Mount tilt.** Is el = 90 deg geocentric nadir, or is the payload already
   tilted toward the pole by a fixed offset?
3. **CoG disk radius R.** What radius should we design to — a single visible
   plume (≲ 2 km), a plant cluster (a few km), or a city-scale set of stacks
   (10–30 km)? This is the number that decides whether 1-axis loses any time.
4. **Tracked target.** Combined centroid of every plume (`R_centroid = Σ w_i r_i`,
   so equal-radius plumes do not grow with N), the currently selected plume
   (origin separation matters), or the covering circle that keeps every plume
   in frame (`R_cover = D + r`)?
5. **In-frame vs on-boresight.** Is science useful anywhere in the chip, or
   must the CoG stay on boresight (the current TRACKING ROI crop assumes we can
   centre the target)?
6. **Altitude.** 400 km, or the current ISS band ~410–420 km?
7. **FOV source.** IMX264 active area (3.23° × 2.70°) or the lens-sheet 2/3"
   format HFOV of 3.36 deg (8.8 mm width)?
8. **Full chip vs inference ROI.** Full 2448×2048 mosaic FOV, or the 256×256
   window the flight software uses today?
9. **Earth rotation / heading.** Spherical non-rotating Earth, or a polar
   latitude pass with ISS inclination 51.6 deg and Earth rotation?
10. **After the elevation stop.** Count the extra seconds while the target
    walks out of the leftover along-track FOV, or stop at the gimbal limit?

## Defaults in this run

See the ASSUMPTIONS block in `analyze.py` and the generated `RESULTS.md`.
The baseline is a 400 km circular orbit, untilted nadir, one-sided 90→30 deg
elevation window, IMX264 active-area FOV, and a CoG disk around an origin that
already sits on the slice centerline.
