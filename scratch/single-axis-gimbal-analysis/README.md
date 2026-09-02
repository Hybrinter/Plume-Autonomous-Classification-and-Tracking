# TEMPORARY ANALYSIS — single-axis gimbal tracking time

This folder is a working study, not flight software. It is not on the CI path.

```text
python3 scratch/single-axis-gimbal-analysis/analyze.py
python3 scratch/single-axis-gimbal-analysis/industrial_latitude.py
```

Geometry: [`RESULTS.md`](RESULTS.md). Worldwide stack inventory:
[`INDUSTRIAL.md`](INDUSTRIAL.md) and [`outputs/`](outputs/).

## Locked assumptions (2026-09-01)

- One-sided elevation window 90→30 deg. **Stop at gimbal limits.**
- Geocentric nadir (≤5 deg offset later).
- At most one plant cluster. Elevation tracks the cluster CoG. Science:
  plume **anywhere in the frame**. Success metric: covering disk `R = D + r`.
  If plants are farther apart than the chip (~±12 km), they are two clusters.
- Current ISS TLE (Celestrak 2026-09-01). Earth rotation + i = 51.63 deg.
  Design pass: **max latitude** (heading due east).
- Most restrictive FOV: IMX264 active area, shrunk by 0.66 % distortion.
- Full-frame inference = full 2×2 band plane (compute note, not FOV).

## Headline (design pass, lat 51.63 deg, h = 433 km)

| Quantity | Value |
| --- | --- |
| Usable FOV | 3.20 deg × 2.68 deg |
| Along-track track time | **124 s** |
| Staring, no gimbal | 3.0 s |
| Peak elevation rate | 0.91 deg/s |
| Nadir lateral half-swath | ±12.1 km |
| 5 km plant cluster, 1-axis vs 2-axis | **0 s lost** |
| Earth-rotation az walk of the origin | 0.007 deg |

At latitudes ≲ 45 deg, Earth rotation plus a 5 km cluster walks the covering
disk out of the chip at the 30 deg stop. One-axis then keeps only the last
tens of seconds. The polar-slice one-axis placement is the high-latitude
case, where that walk is ~0.

## Worldwide stack distribution (Climate TRACE 2025)

| Quantity | Value |
| --- | --- |
| Stack-bearing sources | 20,219 |
| In ISS belt \|lat\| ≤ 51.63° | 92.8% |
| Stack-weighted mean R | 5.0 km |
| Stacks at \|lat\| ≥ 45° (1-axis lossless) | **10%** |
| E[T 1-axis] vs 2-axis, stack-weighted | **50 s vs 121 s (59% lost)** |
| Daily in-swath yield (T × coverage) | **~14×** in favour of 2-axis |

Most stacks sit at 20–40° N. A polar-slice one-axis view keeps the 124 s
window but sees only ~10% of the industry. Working the mid-latitude belt
needs the azimuth axis for Earth-rotation walk, not just for swath.
