# VIS/SWIR dual-branch fusion

**Status:** Proposed. The trade-study models are built in
`tools.ml_models.arch.dual`; no flight module implements fusion. Flight
inference is still single-camera.
**Date:** 2026-10-22

## Objective

Give the classifier and segmentor one VIS tile and one SWIR tile that cover the
same ground, so the VIS view supplies plume outlines and the SWIR view helps
reject cloud. Smoke particles (about 0.1-0.5 um) scatter strongly at visible
wavelengths and weakly at 1.0-1.65 um. Water cloud droplets (about 10 um) stay
bright in both bands.

## Fusion options

| Option | Where the views meet | Cost | Misregistration tolerance |
| --- | --- | --- | --- |
| Early fusion | Input: SWIR resampled to the VIS grid and stacked as a 4th plane | Lowest | Lowest; a shift moves the plane under every filter |
| Mid fusion | Features at output stride 4, after separate branch stems | About the same as early fusion at the stem widths used | Better; the stride-4 features are coarser than the expected shift |
| Late fusion | Outputs of two full networks | About 2x | Highest |

The trade study trains early fusion (`pactnet` and `dilatenet` on the 4-plane
build) and mid fusion (`pactnet_dual` and `dilatenet_dual`) on the same splits,
alongside VIS-only and SWIR-only models.

## Trade-study model

The paired build stores SWIR on the VIS grid, so both fusion types read the
same `(4, H, W)` rows; the dual graphs split the planes internally. Training
rows are 18 x 27 to 39 x 40 VIS samples (one 1.2 km Zenodo scene per row),
which limits the fusion point to output stride 4 or 8.

Two training-only perturbations model flight pairing errors:

- **Registration jitter.** VIS runs at 32 Hz and SWIR at 55 Hz. Ground speed
  under the ISS is about 7 km/s, so a 10 ms frame offset is about 70 m, or
  1-2 VIS samples at the coarse bins. The SWIR plane shifts by up to 1 VIS
  sample per axis.
- **Modality dropout.** Each branch drops with probability 0.1, never both, so
  the model keeps working when one camera is lost or saturated.

The evaluation data is perfectly registered. Jitter can therefore cost the
mid-fusion model a little against early fusion on that data; the comparison
does not measure the misregistration benefit.

## Evaluation

- Same seed-0 splits for every model; the fixed-mix builds for test.
- First results at one seed. Differences of about 0.01 AP are within seed
  noise; a decision needs at least 3 seeds per model.
- The masks are traced in VIS, so segmentation scores favour VIS outlines.

## Flight path after the trade study

- **Native-resolution inputs.** Flight would feed the SWIR tile at its own
  grid (98 x 149 px in pattern A) and the VIS tile at its grid (142 x 216 band
  samples), with the SWIR branch features resampled to the VIS feature grid at
  the fusion point. That needs a two-input dataset row, loader, analysis
  capture, and ONNX export contract.
- **Frame pairing.** A hardware trigger or timestamp match between the two
  cameras, with gimbal-rate compensation, bounds the registration error.
- **Per-camera GSD.** Each branch takes its own GSD encoding.
- **Compute.** The second branch adds stem cost only for mid fusion and about
  2x for late fusion; both must fit the flight inference budget.
- These changes need their own design and decision records before adoption.

## Open items

- Real HD5 spectral response. The simulated SWIR plane is a B8A/B11 blend;
  about 93 percent of the HD5 signal falls where Sentinel-2 has no band.
- Jitter amplitude from measured frame timing.
- Fusion-point choice for flight-size tiles, which are about 5x larger than
  training rows.
