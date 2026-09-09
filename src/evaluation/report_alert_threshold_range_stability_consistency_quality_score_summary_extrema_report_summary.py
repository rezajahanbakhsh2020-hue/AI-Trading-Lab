from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_summary_extrema_report import (
    build_quality_score_summary_extrema_report,
)


def build_quality_score_summary_extrema_report_summary(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a compact summary of the quality extrema report."""

    report = build_quality_score_summary_extrema_report(
        history,
        lower_bound,
        upper_bound,
    )

    positions = {
        "minimum": report["minimum_position"],
        "maximum": report["maximum_position"],
    }

    position_counts = {
        "below_range": 0,
        "inside_range": 0,
        "above_range": 0,
        "none": 0,
    }

    for position in positions.values():
        position_counts[position] += 1

    if report["stable_count"] > report["unstable_count"]:
        stability_status = "stable"
    elif report["stable_count"] < report["unstable_count"]:
        stability_status = "unstable"
    else:
        stability_status = "balanced"

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
        "minimum_position": report["minimum_position"],
        "maximum_position": report["maximum_position"],
        "extrema_spread": report["extrema_spread"],
        "positions": positions,
        "position_counts": position_counts,
        "stability_status": stability_status,
        "details": report["details"],
    }
