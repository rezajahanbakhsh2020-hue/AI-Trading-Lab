from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema import (
    build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema,
)


def build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report(
    history: list[Mapping[str, Any]],
    lower_bound: int,
    upper_bound: int,
) -> dict[str, Any]:
    """Build a structured report from quality-score extrema summaries."""

    report = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema(
            history,
            lower_bound,
            upper_bound,
        )
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
        "improving_metrics": report["improving_metrics"],
        "declining_metrics": report["declining_metrics"],
        "flat_metrics": report["flat_metrics"],
        "positive_change_metrics": report["positive_change_metrics"],
        "negative_change_metrics": report["negative_change_metrics"],
        "improving_metric_count": report["improving_metric_count"],
        "declining_metric_count": report["declining_metric_count"],
        "flat_metric_count": report["flat_metric_count"],
        "positive_change_metric_count": report["positive_change_metric_count"],
        "negative_change_metric_count": report["negative_change_metric_count"],
        "metric_count": report["metric_count"],
        "status_counts": report["status_counts"],
        "changes": report["changes"],
        "ranges": report["ranges"],
        "metric_details": report["metric_details"],
        "details": report["details"],
    }
