from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison_detail import (
    build_quality_score_trend_detail_summary_extrema_comparison_detail,
)


def build_quality_score_trend_detail_summary_extrema_comparison_detail_summary(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a compact summary of detailed quality-score metric comparisons."""

    report = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail(
            history,
            lower_bound,
            upper_bound,
        )
    )

    metric_details = report["metric_details"]

    up_metrics = [
        item["metric"]
        for item in metric_details
        if item["direction"] == "up"
    ]
    down_metrics = [
        item["metric"]
        for item in metric_details
        if item["direction"] == "down"
    ]
    flat_metrics = [
        item["metric"]
        for item in metric_details
        if item["direction"] == "flat"
    ]

    largest_change = report["largest_change"]

    if largest_change > 0:
        overall_change_direction = "up"
    elif largest_change < 0:
        overall_change_direction = "down"
    else:
        overall_change_direction = "flat"

    return {
        "snapshot_count": report["snapshot_count"],
        "lower_bound": report["lower_bound"],
        "upper_bound": report["upper_bound"],
        "quality_trend": report["quality_trend"],
        "stability_trend": report["stability_trend"],
        "consistency_trend": report["consistency_trend"],
        "quality_change": report["quality_change"],
        "stability_change": report["stability_change"],
        "consistency_change": report["consistency_change"],
        "quality_range": report["quality_range"],
        "stability_range": report["stability_range"],
        "consistency_range": report["consistency_range"],
        "quality_peak_snapshot": report["quality_peak_snapshot"],
        "quality_floor_snapshot": report["quality_floor_snapshot"],
        "stability_peak_snapshot": report["stability_peak_snapshot"],
        "stability_floor_snapshot": report["stability_floor_snapshot"],
        "consistency_peak_snapshot": report["consistency_peak_snapshot"],
        "consistency_floor_snapshot": report["consistency_floor_snapshot"],
        "largest_change_metric": report["largest_change_metric"],
        "largest_change": largest_change,
        "overall_change_direction": overall_change_direction,
        "positive_change_metrics": report["positive_change_metrics"],
        "negative_change_metrics": report["negative_change_metrics"],
        "up_metrics": tuple(up_metrics),
        "down_metrics": tuple(down_metrics),
        "flat_metrics": tuple(flat_metrics),
        "up_metric_count": len(up_metrics),
        "down_metric_count": len(down_metrics),
        "flat_metric_count": len(flat_metrics),
        "metric_count": len(metric_details),
        "details": report["details"],
        "metric_details": metric_details,
    }
