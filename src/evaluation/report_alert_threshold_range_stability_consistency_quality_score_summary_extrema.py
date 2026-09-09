from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_summary_detail import (
    build_quality_score_summary_detail,
)


def build_quality_score_summary_extrema(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build extrema information for a stability-consistency quality summary."""

    report = build_quality_score_summary_detail(
        history,
        lower_bound,
        upper_bound,
    )

    details = report["details"]

    if not details:
        minimum_alert_count = None
        maximum_alert_count = None
        minimum_snapshot = None
        maximum_snapshot = None
        alert_count_range = 0
    else:
        minimum_alert_count = min(
            detail["alert_count"]
            for detail in details
        )
        maximum_alert_count = max(
            detail["alert_count"]
            for detail in details
        )

        minimum_snapshot = next(
            detail["snapshot_index"]
            for detail in details
            if detail["alert_count"] == minimum_alert_count
        )
        maximum_snapshot = next(
            detail["snapshot_index"]
            for detail in details
            if detail["alert_count"] == maximum_alert_count
        )

        alert_count_range = (
            maximum_alert_count
            - minimum_alert_count
        )

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
        "minimum_alert_count": minimum_alert_count,
        "maximum_alert_count": maximum_alert_count,
        "minimum_snapshot": minimum_snapshot,
        "maximum_snapshot": maximum_snapshot,
        "alert_count_range": alert_count_range,
        "details": details,
    }
