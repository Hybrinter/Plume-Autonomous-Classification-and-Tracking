# Elevation estimator selection checkpoint

ANALYSIS ONLY. This report does not select or integrate a flight estimator.

The candidates receive the same timestamped propagation, encoder-angle, and
relative-vision history. The residual candidate corrects its residual rate when
the optional predictor reference changes. The joint candidate uses encoder-angle
and relative-vision as separate measurement updates.

## Delayed chronology comparison

| Candidate | Target-rate RMSE (rad/s) | Relative-angle RMSE (rad) | Accepted events |
| --- | ---: | ---: | ---: |
| Corrected residual | 0.00298566 | 0.000630029 | 123 |
| Joint angular | 0.00300776 | 0.000683081 | 123 |

The synthetic sequence covers startup without navigation, navigation appearance,
predictor-reference replacement, navigation loss, nonuniform intervals, encoder
samples, and delayed vision arrivals. It is not an orbital or hardware validation.

## Image-forming exposure sweep

| Candidate | Exposure (us) | Pointing RMSE (rad) | Sharp/detected area (px s) | Area-weighted p90 blur (px) | Worst local blur (px) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CorrectedResidualEstimator | 13 | 0.000332414 | 420.000 | 0.001 | 0.004 |
| CorrectedResidualEstimator | 1000 | 0.000332414 | 420.000 | 0.083 | 0.323 |
| JointAngularEstimator | 13 | 0.000343114 | 420.000 | 0.001 | 0.004 |
| JointAngularEstimator | 1000 | 0.000343114 | 420.000 | 0.083 | 0.323 |

The optical loop integrates a rendered multi-region plume over finite exposures
using simulated gimbal motion and the provisional 0.002636 deg band-plane IFOV.
It reports area-weighted p90 blur, worst local blur, and sharp-and-detected area;
only the rendered centroid reaches the controller model.

## Decision

**UNSELECTED: review paired scenario evidence before flight integration**

Before integration, compare the same candidates with measured encoder timing, camera
exposure metadata, plant uncertainty, plume clipping/deformation, and a physically
calibrated optical model. The present report is an executable test harness and an
explicit review checkpoint, not evidence of flight qualification.
