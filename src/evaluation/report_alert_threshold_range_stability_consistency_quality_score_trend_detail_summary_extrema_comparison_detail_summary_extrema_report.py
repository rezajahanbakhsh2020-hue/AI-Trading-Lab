from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_summary import (
    build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_summary,
)


def build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a final structured report from the extrema summary."""

    summary = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_summary(
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

    metric_status = {}

    for metric in metrics:
        change = summary[f"{metric}_change"]

        if change > 0:
            direction = "up"
        elif change < 0:
            direction = "down"
        else:
            direction = "flat"

        metric_status[metric] = {
            "change": change,
            "direction": direction,
            "range": summary["range_spread"][metric],
            "peak_snapshot": summary["peak_snapshots"][metric],
            "floor_snapshot": summary["floor_snapshots"][metric],
            "range_rank": summary["range_rank"][metric],
        }

    return {
        "snapshot_count": summary["snapshot_count"],
        "lower_bound": summary["lower_bound"],
        "upper_bound": summary["upper_bound"],
        "quality_trend": summary["quality_trend"],
        "stability_trend": summary["stability_trend"],
        "consistency_trend": summary["consistency_trend"],
        "quality_change": summary["quality_change"],
        "stability_change": summary["stability_change"],
        "consistency_change": summary["consistency_change"],
        "largest_change_metric": summary["largest_change_metric"],
        "largest_change": summary["largest_change"],
        "overall_change_direction": summary["overall_change_direction"],
        "widest_metric": summary["widest_metric"],
        "widest_range": summary["widest_range"],
        "range_order": summary["range_order"],
        "range_rank": summary["range_rank"],
        "range_spread": summary["range_spread"],
        "peak_snapshots": summary["peak_snapshots"],
        "floor_snapshots": summary["floor_snapshots"],
        "metric_extrema": summary["metric_extrema"],
        "metric_status": metric_status,
        "details": summary["details"],
        "metric_details": summary["metric_details"],
    }
