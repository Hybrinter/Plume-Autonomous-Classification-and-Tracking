# Single-axis vs dual-axis gimbal

TEMPORARY ANALYSIS. Not flight software. Design study for dropping
the azimuth gimbal axis. Shared geometry lives in `analysis.lib`;
this folder holds generated reports, CSV, and local inventory downloads.

```text
uv run python -m analysis.studies.single_axis_vs_dual_axis_gimbal geometry
uv run python -m analysis.studies.single_axis_vs_dual_axis_gimbal industry
uv run python -m analysis.studies.single_axis_vs_dual_axis_gimbal all
```

Geometry: [`RESULTS.md`](RESULTS.md). Worldwide stack inventory:
[`INDUSTRIAL.md`](INDUSTRIAL.md) and [`outputs/`](outputs/).

## Physical setup

- **Fixed origin.** Elevation tracks the stack, or the cluster centroid of
  stacks. Wind can swing the visible plume around that point during the
  ~120 s window. Tracking plume CoG would add lateral walk this study
  does not model.
- **Covering disk.** Unknown wind azimuth makes a disk of radius **L**
  around each stack. Cluster covering radius is `R = D + L`, with D the
  haversine plant span. That is a possibility set, not a smoke-filled blob.
- **L = 2 km** is a **conservative visible-length envelope**
  (cooling-tower photo climatologies: typical 0.3-0.8 km, winter often
  >0.9 km, ~90-95th percentile ~1.5-3 km). Not a Mommert-chip measurement.
  Not typical length.
- **P(visible)** = fraction of that disk in the FOV (unknown-wind geometry).
  Plume volume / Gaussian ribbon (~5% disk fill) is occupancy/SNR only.
  Do not multiply the two.
- **No locked design R.** Operational R is per cluster (`r_cover_km`) and,
  for lat tables, stack-weighted mean `R(|lat|) = D(|lat|) + L`.
- **Camera/lens.** BFS-U3-50S5, Sony IMX264, 3.45 um, 2448x2048,
  2/3-inch, global shutter; 150 mm athermal, 0.66% distortion. Catalog
  2/3-inch HFOV 3.36 deg is a check. 2x2 mosaic => band plane
  1224x1024, IFOV x2. Ignore lens-catalog 2.74 um.
- **Two slew caps.** Imaging rewind vs ground is the 1-pixel / 1 ms
  **band-plane** gate (current imaging gate). Hardware cap
  **10 deg/s** is not for science frames. SIL
  `ifov=0.02` is not this optic.
- **Reacquire** starts when leftover window opens (hunt during slew).
  Do not wait until elevation is back at 30 deg. One-axis hunt is
  elevation-only (FOV ribbon). Two-axis hunt would use the gimbal box.
  Mean encounter is Poisson / mean-spacing, ocean-averaged in a lat band,
  conditional on the ISS belt -- not a city-corridor nearest neighbour.

## Headline (design pass, lat 51.63 deg, h = 433 km)

| Quantity | Value |
| --- | --- |
| Camera / lens | BFS-U3-50S5 + 150 mm, Sony IMX264 |
| Usable FOV | 3.20 deg x 2.68 deg |
| Band plane | 1224 x 1024 (IFOV x2) |
| Along-track track time | **124 s** |
| Staring, no gimbal | 3.0 s |
| Peak elevation rate | 0.91 deg/s |
| Imaging rewind (current gate) | **1.72 deg/s** |
| Scene-relative smear limit | 2.64 deg/s |
| Hardware slew cap | 10 deg/s |
| Nadir lateral half-swath | +/-12.1 km |
| 2-axis +/-10 deg half-swath | +/-76 km |
| Earth-rotation az walk of the origin | 0.007 deg |

At the design pass the origin stays in the chip. There is no locked
design R. Mid-latitude loss is set by Earth rotation plus R(|lat|)
from the inventory, not by a 5 km placeholder.

## Worldwide stack distribution (Climate TRACE 2025)

| Quantity | Value |
| --- | --- |
| ISS-belt stack fraction | 92.8% |
| Stack-weighted mean R | 4.98 km (inventory, not design) |
| Stacks at |lat| >= 45 deg | **10%** |
| E[T 1-axis] vs 2-axis, stack-weighted | **50 s vs 121 s (59% lost)** |
| After leftover hunt at omega_img | **63 s vs 121 s (48% lost)** |
| Daily in-swath yield (T x coverage) | **~14x** in favour of 2-axis |

| |lat| (deg) | D (km) | R (km) | singleton % | d_char n>=2 (km) |
| --- | --- | --- | --- | --- |
| 0-10.0 | 1.90 | 3.90 | 50% | 3.72 |
| 10-20.0 | 2.16 | 4.16 | 50% | 4.16 |
| 20-30.0 | 2.87 | 4.87 | 41% | 4.62 |
| 30-40.0 | 3.51 | 5.51 | 35% | 4.85 |
| 40-45.0 | 2.96 | 4.96 | 38% | 4.57 |
| 45-51.6 | 2.31 | 4.31 | 46% | 4.27 |

Most stacks sit at 20-40 N. A polar-slice one-axis view keeps the
124 s window but sees only ~10% of the industry.
Working the mid-latitude belt needs the azimuth axis for Earth-rotation
walk, not just for swath. Leftover hunt during the slew recovers some
one-axis time only if another cluster sits on the FOV ribbon.

