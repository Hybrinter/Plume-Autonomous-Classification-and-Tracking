# VIS/SWIR common-FOV tiling (pattern A)

**Status:** Proposed for the VIS/SWIR trade study. Not as-built.
No flight module implements this; flight tiling is still the single-camera
8 x 8 grid in `flight.payload.preprocess.tiling`.
**Date:** 2026-10-21 (50 mm SWIR lens; replaces the 40 mm pattern A' of 2026-10-19)

## Objective

Define one tile grid that both cameras sample, so a fused model sees
the same ground in every VIS tile and its paired SWIR tile. The trade study
uses the grid to generate paired training tiles and to compare VIS-only,
SWIR-only, and fused models.

## Candidate hardware

| Item | VIS | SWIR |
| --- | --- | --- |
| Camera | JAI GOX-8901C-USB (Sony IMX267, Bayer) | Attollo Phoenix HD5 (InGaAs, mono) |
| Sensor | 4096 x 2160 px, 3.45 um | 1280 x 1024 px, 5 um |
| Readout used | 3840 x 2160 px centered ROI | full sensor |
| Spectral band | 400-700 nm (lens coating 425-675 nm) | 1.0-1.65 um (datasheet) or 0.9-1.7 um (product page); unresolved |
| Lens | 100 mm | 50 mm (selected; sweep 100/75/50/40/35 mm), f-number not yet known |
| Sample pitch | 6.9 um per color band (2 x 2 Bayer) | 5 um |
| Angle per sample | 69 urad | 100 urad |

## Sample-pitch convention

The VIS sensor samples each color every second pixel. One VIS band sample is
6.9 um and subtends 69 urad at 100 mm. GSD values computed with 69 urad are
metres per band sample, and must be multiplied by band-sample counts
(1920 x 1080 for the 3840 x 2160 ROI), not sensor-pixel counts. Mixing band GSD
with sensor-pixel counts overstates the VIS footprint by 2x per axis and 4x in
area. A tile size may be quoted in sensor pixels (2 x band samples) for a
demosaicked image; its effective resolution is still the band pitch.

## Common field of view

For a camera with `n` samples of pitch `p` behind focal length `f`, the field
of view along one axis is `n * p / f`. The common field of view is the
per-axis minimum of the two cameras:

    theta_common = min(n_vis * p_vis / f_vis, n_swir * p_swir / f_swir)

Each camera keeps `floor(theta_common / ifov)` samples per axis, centered, and
discards the rest. At nominal focal lengths (VIS 100 mm, SWIR 50 mm):

| Axis | VIS FOV (ROI) | SWIR FOV | Common FOV | VIS kept (band samples) | SWIR kept (px) |
| --- | --- | --- | --- | --- | --- |
| Horizontal (cross-track) | 132.5 mrad | 128.0 mrad | 128.0 mrad (SWIR-limited) | 1855 | 1280 |
| Vertical (along-track) | 74.5 mrad | 102.4 mrad | 74.5 mrad (VIS-limited) | 1080 | 745 |

Horizontally the SWIR field is narrower than the VIS ROI, so the outer
VIS columns have no SWIR pair. Vertically the SWIR field contains the VIS
ROI. SWIR keeps about 73 percent of its sensor. At the 45 deg along-track
reference look (440 km orbit, 2 km plume height), the common footprint is
about 82 x 73 km.

## Pattern A

Pattern A is a 13 x 5 grid of rectangular tiles whose VIS and SWIR edges
subtend the same angle within a fraction of a SWIR pixel.

| Quantity | VIS | SWIR |
| --- | --- | --- |
| Grid (cols x rows) | 13 x 5 = 65 tiles | 13 x 5 = 65 tiles |
| Tile size | 142 x 216 band samples (284 x 432 sensor px) | 98 x 149 px |
| Tile angle | 9.798 x 14.904 mrad | 9.800 x 14.900 mrad |
| Grid extent | 1846 x 1080 band samples (3692 x 2160 sensor px) | 1274 x 745 px |
| Centered crop from full sensor | 202 sensor columns per side (74 inside the ROI), no rows | 3 columns per side, 139/140 rows top/bottom |
| GSD, nadir | about 30 m per band sample | about 44 m |
| GSD, 45 deg reference | about 44 x 68 m | about 64 x 98 m |
| Tile footprint, nadir | about 4.3 x 6.5 km | same |
| Tile footprint, 45 deg reference | about 6.3 x 14.6 km | same |

Grid coverage of the common field of view is 99.5 percent. Accumulated edge
misalignment across the grid is at most 0.26 SWIR px (0.20 px vertically). The
VIS crop offsets are even numbers of sensor pixels, which keeps the same Bayer
phase at every tile origin. The tile count stays close to the current 64.
Tiles are taller than wide (aspect 1.52) along-track.

## Tile selection rule

The grid search treats each axis independently. For a VIS tile of `a` band
samples, the SWIR tile is `c = round(a * ifov_vis / ifov_swir)` px. The grid
count is `n = min(floor(N_vis / a), floor(N_swir / c))`, where `N` is the kept
sample count. Accumulated drift is `n * |a * ifov_vis - c * ifov_swir| / ifov_swir`
SWIR px. Pattern A maximizes area coverage subject to drift <= 0.5 px, a tile
count near the current 64, and tile aspect ratio <= 2.

Tile dimensions cannot be multiples of 8 in both cameras at a 0.69 sample
ratio. Models that need divisible sizes pad internally.

## Focal-length sweep

The trade study evaluates SWIR focal lengths of 100, 75, 50, 40, and 35 mm,
all with the 3840 x 2160 VIS ROI. Each focal length has its own common field of
view and best grid:

| SWIR lens | SWIR nadir GSD | Kept VIS band / SWIR px | Best grid | Coverage of common FOV |
| --- | --- | --- | --- | --- |
| 100 mm | about 22 m | 927 x 742 / 1280 x 1024 | 13 x 8 = 104 | 98.8 percent |
| 75 mm | about 29 m | 1236 x 989 / 1280 x 1024 | 6 x 7 = 42 | 97.9 percent |
| 50 mm | about 44 m | 1855 x 1080 / 1280 x 745 | 13 x 5 = 65 (pattern A) | 99.5 percent |
| 40 mm | about 55 m | 1920 x 1080 / 1059 x 596 | 9 x 7 = 63 (former pattern A') | 99.2 percent |
| 35 mm | about 63 m | 1920 x 1080 / 927 x 521 | 11 x 4 = 44 | 99.3 percent |

At 100 mm the common field of view is about one third of the VIS frame. At
40 mm and shorter the whole VIS ROI is covered, and a shorter lens only
coarsens the SWIR GSD.

## Limits and open items

- **Nominal geometry only.** A 1 percent focal-length error moves the last
  tile edge by about 10 SWIR px. A flight grid must be computed from measured
  focal lengths, measured distortion, and measured boresight offset, with a
  1-2 percent crop margin for boresight tolerance. At 50 mm the SWIR
  field extends past the VIS ROI by about 14 mrad per side vertically, but
  it is the horizontal limit. Horizontal boresight error is absorbed by the
  VIS crop, which has about 7 mrad per side of unused sensor.
  Implementations take the measured angle per sample as an input; they do not
  hard-code pattern A pixel counts.
- **Early fusion resamples anyway.** For a stacked VIS+SWIR input, SWIR is
  resampled onto the VIS tile grid, so integer alignment is a convenience.
  Native-resolution SWIR tiles (mid or late fusion) rely on it more. At
  50 mm one SWIR pixel spans about 1.45 VIS band samples.
- **Training-tile scale.** A pattern A tile spans 4-15 km. Zenodo 4250706
  scenes span 1.2 km, so a resampled Zenodo scene fills a small part of one
  tile. The dataset study reports this gap.
- **SWIR 50 mm lens.** No lens part number or f-number is known; the
  training build assumes f/2.8. The lens must cover
  the 8.2 mm HD5 diagonal across the band chosen below.
- **VIS lens format.** The VIS lens must cover the 1 in class IMX267
  (16 mm diagonal). The current flight lens is specified for 1/1.8 in sensors.
- **SWIR band edges.** The HD5 cutoff decides which Sentinel-2 bands feed the
  simulated SWIR channel.
- **Frame timing.** VIS runs at 32 Hz and SWIR at 55 Hz; pairing needs a
  frame-selection rule.
- **GSD convention for conditioning.** The paired training build records
  metres per VIS band sample (6.9 um pitch). The current flight sensor
  configuration derives its 15.87 m reference GSD from the 3.45 um mosaic
  pitch. A flight adoption must use the band-sample convention, or the
  GSD fed to the model is half the training value.

## Flight impact after adoption

Adoption would change the inference grid configuration, `flight.payload.preprocess.tiling`
(per-camera grids from one angular grid), the footprint and GSD encoding
(per-camera GSD), the sensor configuration (two sensors), and the model export
contract (paired inputs). Those changes need their own design and decision
records after the purchase decision.
