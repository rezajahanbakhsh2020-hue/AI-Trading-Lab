from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_summary_extrema import (
    build_quality_score_summary_extrema,
)


def build_quality_score_summary_extrema_report(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a complete report for stability-consistency quality extrema."""

    report = build_quality_score_summary_extrema(
        history,
        lower_bound,
        upper_bound,
    )

    if report["minimum_alert_count"] is None:
        minimum_position = "none"
    elif report["minimum_alert_count"] < lower_bound:
        minimum_position = "below_range"
    elif report["minimum_alert_count"] > upper_bound:
        minimum_position = "above_range"
    else:
        minimum_position = "inside_range"

    if report["maximum_alert_count"] is None:
        maximum_position = "none"
    elif report["maximum_alert_count"] < lower_bound:
        maximum_position = "below_range"
    elif report["maximum_alert_count"] > upper_bound:
        maximum_position = "above_range"
    else:
        maximum_position = "inside_range"

    if report["alert_count_range"] == 0:
        extrema_spread = "flat"
    elif (
        report["minimum_alert_count"] is not None
        and report["maximum_alert_count"] is not None
        and report["minimum_alert_count"] >= lower_bound
        and report["maximum_alert_count"] <= upper_bound
    ):
        extrema_spread = "inside_range"
    else:
        extrema_spread = "crosses_range"

    return {
        "snapshot_count": report["snapshot_count"],
        "lower_bound": report["lower_bound"],
        "upper_bound": report["upper_bound"],
        "stability_ratio": report["stability_ratio"],
        "consistency_ratio": report["consistency_ratio"],
        "quality_score": report["quality_score"],
        "quality_percentage": report["quality_percentage"],
        "stable_count": report["stable_count"],
        "unstable_count": report["unstable_count"],
        "minimum_alert_count": report["minimum_alert_count"],
        "maximum_alert_count": report["maximum_alert_count"],
        "minimum_snapshot": report["minimum_snapshot"],
        "maximum_snapshot": report["maximum_snapshot"],
        "alert_count_range": report["alert_count_range"],
        "minimum_position": minimum_position,
        "maximum_position": maximum_position,
        "extrema_spread": extrema_spread,
        "details": report["details"],
    }
