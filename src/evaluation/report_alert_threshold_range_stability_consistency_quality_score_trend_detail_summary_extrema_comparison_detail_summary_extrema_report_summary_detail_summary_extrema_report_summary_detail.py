from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report_summary import (
    build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report_summary,
)


def build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report_summary_detail(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build detailed metric information from the quality-score report summary."""

    report = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report_summary(
            history,
            lower_bound,
            upper_bound,
        )
    )

    metric_details = report["metric_details"]

    details: list[dict[str, Any]] = []

    for item in metric_details:
        status = item["status"]
        direction = status["direction"]
        change = item["change"]
        metric_range = item["range"]

        if direction == "up":
            change_class = "positive"
        elif direction == "down":
            change_class = "negative"
        else:
            change_class = "neutral"

        details.append(
            {
                "metric": item["metric"],
                "change": change,
                "direction": direction,
                "change_class": change_class,
                "range": metric_range,
                "peak_snapshot": status["peak_snapshot"],
                "floor_snapshot": status["floor_snapshot"],
                "range_rank": status["range_rank"],
                "is_improving": item["is_improving"],
                "is_declining": item["is_declining"],
                "is_flat": item["is_flat"],
            }
        )

    improving_metrics = tuple(
        item["metric"]
        for item in details
        if item["is_improving"]
    )
    declining_metrics = tuple(
        item["metric"]
        for item in details
        if item["is_declining"]
    )
    flat_metrics = tuple(
        item["metric"]
        for item in details
        if item["is_flat"]
    )

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
        "largest_change_metric": report["largest_change_metric"],
        "largest_change": report["largest_change"],
        "overall_change_direction": report["overall_change_direction"],
        "widest_metric": report["widest_metric"],
        "widest_range": report["widest_range"],
        "largest_change_abs_metric": report["largest_change_abs_metric"],
        "largest_change_abs": report["largest_change_abs"],
        "smallest_change_abs_metric": report["smallest_change_abs_metric"],
        "smallest_change_abs": report["smallest_change_abs"],
        "maximum_change_metric": report["maximum_change_metric"],
        "maximum_change": report["maximum_change"],
        "minimum_change_metric": report["minimum_change_metric"],
        "minimum_change": report["minimum_change"],
        "change_range": report["change_range"],
        "narrowest_metric": report["narrowest_metric"],
        "narrowest_range": report["narrowest_range"],
        "range_range": report["range_range"],
        "improving_metrics": improving_metrics,
        "declining_metrics": declining_metrics,
        "flat_metrics": flat_metrics,
        "positive_change_metrics": report["positive_change_metrics"],
        "negative_change_metrics": report["negative_change_metrics"],
        "improving_metric_count": len(improving_metrics),
        "declining_metric_count": len(declining_metrics),
        "flat_metric_count": len(flat_metrics),
        "positive_change_metric_count": report[
            "positive_change_metric_count"
        ],
        "negative_change_metric_count": report[
            "negative_change_metric_count"
        ],
        "metric_count": report["metric_count"],
        "status_counts": report["status_counts"],
        "direction_counts": report["direction_counts"],
        "changes": report["changes"],
        "ranges": report["ranges"],
        "metric_details": details,
        "details": report["details"],
    }
