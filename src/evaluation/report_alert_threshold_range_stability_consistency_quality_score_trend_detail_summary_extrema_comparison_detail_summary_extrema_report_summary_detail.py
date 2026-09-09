from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary import (
    build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary,
)


def build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build per-metric detail records from the report summary."""

    report = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary(
            history,
            lower_bound,
            upper_bound,
        )
    )

    metrics = (
        "quality",
        "stability",
        "consistency",
    )

    metric_details: list[dict[str, Any]] = []

    for metric in metrics:
        change = report[f"{metric}_change"]
        status = report["metric_status"][metric]
        extrema = report["metric_extrema"][metric]

        if change > 0:
            direction = "up"
        elif change < 0:
            direction = "down"
        else:
            direction = "flat"

        metric_details.append(
            {
                "metric": metric,
                "change": change,
                "direction": direction,
                "range": extrema["range"],
                "peak_snapshot": extrema["peak_snapshot"],
                "floor_snapshot": extrema["floor_snapshot"],
                "range_rank": report["metric_extrema"][metric].get(
                    "range_rank",
                    report.get("range_rank", {}).get(metric),
                ),
                "is_widest": metric == report["widest_metric"],
                "is_largest_change": metric == report["largest_change_metric"],
                "is_improving": metric in report["improving_metrics"],
                "is_declining": metric in report["declining_metrics"],
                "is_flat": metric in report["flat_metrics"],
                "status": status,
            }
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
        "improving_metrics": report["improving_metrics"],
        "declining_metrics": report["declining_metrics"],
        "flat_metrics": report["flat_metrics"],
        "improving_metric_count": report["improving_metric_count"],
        "declining_metric_count": report["declining_metric_count"],
        "flat_metric_count": report["flat_metric_count"],
        "metric_details": metric_details,
        "details": report["details"],
    }
