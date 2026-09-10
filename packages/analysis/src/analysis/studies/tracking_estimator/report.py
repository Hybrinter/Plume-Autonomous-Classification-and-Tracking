"""Selection-checkpoint report writer for the estimator comparison study."""

from __future__ import annotations

from pathlib import Path

from analysis.studies.tracking_estimator.scenarios import (
    ComparisonResult,
    OpticalMetrics,
    run_estimator_comparison,
    run_optical_exposure_sweep,
)


def write_selection_checkpoint(
    path: Path,
    comparison: ComparisonResult | None = None,
    optical: tuple[OpticalMetrics, ...] | None = None,
) -> Path:
    """Write a reviewable report without selecting or modifying a flight estimator."""
    result = comparison if comparison is not None else run_estimator_comparison()
    optical_results = optical if optical is not None else run_optical_exposure_sweep()
    lines: list[str] = []
    append = lines.append
    append("# Elevation estimator selection checkpoint")
    append("")
    append("ANALYSIS ONLY. This report does not select or integrate a flight estimator.")
    append("")
    append("The candidates receive the same timestamped propagation, encoder-angle, and")
    append("relative-vision history. The residual candidate corrects its residual rate when")
    append("the optional predictor reference changes. The joint candidate uses encoder-angle")
    append("and relative-vision as separate measurement updates.")
    append("")
    append("## Delayed chronology comparison")
    append("")
    append("| Candidate | Target-rate RMSE (rad/s) | Relative-angle RMSE (rad) | Accepted events |")
    append("| --- | ---: | ---: | ---: |")
    append(
        "| Corrected residual | "
        f"{result.residual.target_rate_rmse_rad_s:.6g} | "
        f"{result.residual.relative_angle_rmse_rad:.6g} | {result.residual.accepted_events} |"
    )
    append(
        "| Joint angular | "
        f"{result.joint.target_rate_rmse_rad_s:.6g} | "
        f"{result.joint.relative_angle_rmse_rad:.6g} | {result.joint.accepted_events} |"
    )
    append("")
    append("The synthetic sequence covers startup without navigation, navigation appearance,")
    append("predictor-reference replacement, navigation loss, nonuniform intervals, encoder")
    append("samples, and delayed vision arrivals. It is not an orbital or hardware validation.")
    append("")
    append("## Image-forming exposure sweep")
    append("")
    append(
        "| Candidate | Exposure (us) | Pointing RMSE (rad) | Sharp/detected area (px s) | "
        "Area-weighted p90 blur (px) | Worst local blur (px) |"
    )
    append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for item in optical_results:
        append(
            f"| {item.estimator_name} | {item.exposure_us:g} | {item.pointing_rmse_rad:.6g} | "
            f"{item.retained_sharp_detected_area_px_s:.3f} | "
            f"{item.area_weighted_p90_blur_px:.3f} | {item.worst_local_blur_px:.3f} |"
        )
    append("")
    append("The optical loop integrates a rendered multi-region plume over finite exposures")
    append("using simulated gimbal motion and the provisional 0.002636 deg band-plane IFOV.")
    append("It reports area-weighted p90 blur, worst local blur, and sharp-and-detected area;")
    append("only the rendered centroid reaches the controller model.")
    append("")
    append("## Decision")
    append("")
    append(f"**{result.selection_status}**")
    append("")
    append("Before integration, compare the same candidates with measured encoder timing, camera")
    append("exposure metadata, plant uncertainty, plume clipping/deformation, and a physically")
    append("calibrated optical model. The present report is an executable test harness and an")
    append("explicit review checkpoint, not evidence of flight qualification.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
